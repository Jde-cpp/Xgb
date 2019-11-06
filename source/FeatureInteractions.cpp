#include "stdafx.h"
#include "FeatureInteractions.h"
#include "/home/duffyj/code/libraries/LightGBM/include/LightGBM/tree.h"

namespace Jde::AI::Xgb
{
	MyGbdt::MyGbdt( const fs::path& path )noexcept(false)
	{
		if( !Boosting::LoadFileToBoosting(this, path.c_str()) )
			throw IOException( "Could not load {}.", path );
	}

	typedef std::vector<std::string> XgbModelDump;

	XgbModelDump XgbModelParser::ReadModelDump(const std::string& file_path, int max_trees) 
	{
		XgbModelDump dump;
		if (max_trees == 0) return dump;

		std::ifstream ifs(file_path);
		if (!ifs) {
			std::cerr << "could not read file " << file_path << std::endl;
			return dump;
		}
		std::string line;
		std::ostringstream str_buffer;
		while (std::getline(ifs, line)) {
			if (line.find_first_of("booster") == 0) {
				if (str_buffer.tellp() > 0) {
				if (--max_trees == 0) break;
				dump.push_back(str_buffer.str());
					str_buffer = std::ostringstream();
				str_buffer.clear();
				str_buffer.str({});
				}
				continue;
			}
			str_buffer << line << std::endl;
		}
		dump.push_back(str_buffer.str());
		return dump;
	}

	void ConstructXgbTree(XgbTreePtr tree, const XgbNodeList& nodes) 
	{
		if (tree->root.left_child != -1) 
		{
			tree->left = std::make_shared<XgbTree>( XgbTree(nodes.at(tree->root.left_child)) );
			ConstructXgbTree( tree->left, nodes );
		}
		if (tree->root.right_child != -1) 
		{
			tree->right = std::make_shared<XgbTree>(XgbTree(nodes.at(tree->root.right_child)));
			ConstructXgbTree(tree->right, nodes);
		}
	}

	XgbModel XgbModelParser::GetXgbModelFromDump(const std::string& file_path, int max_tress) 
	{
		return XgbModelParser::GetXgbModelFromDump( XgbModelParser::ReadModelDump( file_path, max_tress ), max_tress );
	}

	XgbModel XgbModelParser::GetXgbModelFromDump(const XgbModelDump& dump, int max_tress) 
	{
		int ntrees = static_cast<int>(dump.size());
		if ((max_tress < ntrees) && (max_tress >= 0)) 
			ntrees = max_tress;

		XgbModel xgb_model(ntrees);
		XgbNodeLists xgb_node_lists = {};

		for (int i = 0; i < ntrees; ++i) {
			xgb_node_lists.push_back(XgbNodeList{});
		}

		for (int i = 0; i < ntrees; ++i) 
		{
			std::istringstream iss(dump[i]);
			std::string line;
			auto nodes = &xgb_node_lists[i];
			while (std::getline(iss, line)) 
			{
				auto&& xgb_tree_node = ParseXgbTreeNode(&line);
				(*nodes)[xgb_tree_node.number] = std::move(xgb_tree_node);
			}
		}

		for (int i = 0; i < ntrees; ++i) 
		{
			auto&& tree = std::make_shared<XgbTree>(XgbTree(xgb_node_lists[i][0], i));
			ConstructXgbTree( tree, xgb_node_lists[i] );
			xgb_model.trees[i] = std::move(tree);
		}

		return xgb_model;
	}

	XgbTreeNode::XgbTreeNode( const LightGBM::Tree& tree, uint number ):
		Number{number},
		Feature{ tree.split_feature(number) },
		Gain{tree.split_gain(number) },
	{
		double Cover{0.0};
		uint LeftChild{-1};
		uint RightChild{-1};
		double SplitValue{0};
		double LeafValue{0};
		bool IsLeaf{false};
	}

	void XgbModel::CollectFeatureInteractions( LightGBM::Tree& tree, InteractionPath& cfi, double current_gain, double current_cover, double path_proba, int depth, int deepening, FeatureInteractions* tfis, PathMemo* memo ) 
	{
		if( tree.num_leaves()==1 || depth == _maxTreeDepth )
			return;

		cfi.push_back( tree->root );
		current_gain += tree->root.gain;
		current_cover += tree->root.cover;

		auto ppl = path_proba * (tree->left->root.cover / tree->root.cover);
		auto ppr = path_proba * (tree->right->root.cover / tree->root.cover);

		FeatureInteraction fi( cfi, current_gain, current_cover, path_proba, depth, 1 );

		if( (depth < max_deepening_) || (max_deepening_ < 0) ) 
		{
			InteractionPath ipl{};
			InteractionPath ipr{};

			CollectFeatureInteractions( tree->left, &ipl, 0, 0, ppl, depth + 1, deepening + 1, tfis, memo );
			CollectFeatureInteractions( tree->right, &ipr, 0, 0, ppr, depth + 1, deepening + 1, tfis, memo );
		}

		auto path = FeatureInteraction::InteractionPathToStr( cfi, true );

		if (tfis->find(fi.name) == tfis->end()) 
		{
			(*tfis)[fi.name] = fi;
			memo->insert(path);
		} 
		else 
		{
			if( memo->count(path) ) 
				return;
			memo->insert(path);
			auto tfi = &(*tfis)[fi.name];
			tfi->gain += current_gain;
			tfi->cover += current_cover;
			tfi->fscore += 1;
			tfi->w_fscore += path_proba;
			tfi->avg_w_fscore = tfi->w_fscore / tfi->fscore;
			tfi->avg_gain = tfi->gain / tfi->fscore;
			tfi->expected_gain += current_gain * path_proba;
		}

		if( static_cast<int>( cfi.size() ) - 1 == max_interaction_depth_ )
			return;

		InteractionPath ipl{ cfi };
		InteractionPath ipr{ cfi };

		CollectFeatureInteractions( tree->left, ipl, current_gain, current_cover, ppl, depth + 1, deepening, tfis, memo );
		CollectFeatureInteractions( tree->right, ipr, current_gain, current_cover, ppr, depth + 1, deepening, tfis, memo );
	}

	FeatureInteractions XgbModel::GetFeatureInteractions(int max_interaction_depth, int max_tree_depth, int max_deepening) 
	{
		max_interaction_depth_ = max_interaction_depth;
		_maxTreeDepth = max_tree_depth;
		max_deepening_ = max_deepening;

		std::vector<FeatureInteractions> trees_feature_interactions( ntrees );

		for (int i = 0; i < ntrees; ++i) {
			FeatureInteractions tfis{};
			InteractionPath cfi{};
			PathMemo memo{};
			CollectFeatureInteractions(trees[i], &cfi, 0, 0, 1, 0, 0, &tfis, &memo);
			trees_feature_interactions[i] = tfis;
		}

		FeatureInteractions fis;

		for (int i = 0; i < ntrees; ++i) 
			FeatureInteraction::Merge(&fis, trees_feature_interactions[i]);

		return fis;
	}


	std::vector<std::string> GetFeatureInteractions( const xgboost::Learner& learner, uint max_fi_depth, uint max_tree_depth, uint max_deepening, uint ntrees, const fs::path& featureMapPath )
	{
		std::vector<std::string> featureInteractions;
		xgboost::FeatureMap featureMap;
		if (strchr(fmap, '|') != NULL) {
			int fnum = 0;
			const char* ftype = "q";
			for (auto feat : StringUtils::split(fmap, '|')) {
			featureMap.PushBack(fnum++, feat.c_str(), ftype);
			}
		} else if (*fmap != '\0') {
			try {
			std::unique_ptr<dmlc::Stream> fs(dmlc::Stream::Create(fmap, "r"));
			dmlc::istream is(fs.get());
			featureMap.LoadText(is);
			}
			catch (...) {
			LOG(CONSOLE) << "Warning: unable to read feature map: \"" << fmap << "\", "
				"feature names wont be mapped";
			}
		}
		auto dump = learner.DumpModel( featureMap, true, "text" );
		if (dump.size() == 0) {
			return featureInteractions;
		}
		if (dump[0].find_first_of("bias") == 0) {
			return featureInteractions;
		}
		auto model = xgbfi::XgbModelParser::GetXgbModelFromDump( dump, ntrees );
		auto fi = model.GetFeatureInteractions( max_fi_depth, max_tree_depth, max_deepening );
		for (auto kv : fi) {
			featureInteractions.push_back(static_cast<std::string>(kv.second));
		}
		return featureInteractions;
	}

}