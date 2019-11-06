#pragma once
#include "/home/duffyj/code/libraries/LightGBM/src/boosting/gbdt.h"
namespace Jde::AI::Xgb
{
	struct MyGbdt : LightGBM::GBDT
	{
		MyGbdt( const fs::path& path )noexcept(false);
	};
	struct XgbModel
	{
		uint _maxTreeDepth;
	}
	std::vector<std::string> GetFeatureInteractions( const xgboost::Learner& learner, uint max_fi_depth, uint max_tree_depth, uint max_deepening, uint ntrees, const fs::path& featureMapPath );

	struct XgbTreeNode
	{
		XgbTreeNode( const LightGBM::Tree& tree, int number );
		int Number{0};
		uint Feature;
		double Gain{0.0};
		double Cover{0.0};
		uint LeftChild{-1};
		uint RightChild{-1};
		double SplitValue{0};
		double LeafValue{0};
		bool IsLeaf{false};
  	};	
  	typedef std::vector<XgbTreeNode> InteractionPath;
}