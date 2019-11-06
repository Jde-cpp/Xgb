#include "stdafx.h"
#include "Booster.h"
#include <string.h>
#include "Dataset.h"
#include "XgbBoosterParams.h"
//#include "../IDataset.h"
#include "../../DecisionTree/source/IBoosterParams.h"
#include "../../DecisionTree/source/EFeatureImportance.h"
#include "../../DecisionTree/source/TreeNode.h"
#define var const auto


namespace Jde::AI::Dts::Xgb
{
	//static char* Booster::FileSuffix = ".model";
	Booster::Booster( const Jde::AI::Dts::IBoosterParams& params, sp<const IDataset>& pDS, sp<const IDataset> pValidation )noexcept(false):
		TrainRowCount{ pDS->RowCount },
		ValidationRowCount{ pValidation ? pValidation->RowCount : 0 },
		_pValidation{ std::dynamic_pointer_cast<const Dataset>(pValidation) },
		_pTrain{ std::dynamic_pointer_cast<const Dataset>(pDS) }
	{
		//_pValidation = std::dynamic_pointer_cast<const Dataset>( pValidation );
		//_pTrain = std::dynamic_pointer_cast<const Dataset>(pDS);
		//const Dataset& ds2 = dynamic_cast<const Dataset&>( ds );
		//_pTrain = ds2.Handle();
		var handle = _pTrain->Handle();
		const DMatrixHandle dmats[1]{handle};
		CALL( XGBoosterCreate(dmats, 1, &_handle), "XGBoosterCreate" );

		var& paramMap = params.Parameters();
		for( var& [name, pValue] : paramMap )
			CALL( XGBoosterSetParam(_handle, name.c_str(), pValue->InitialString().c_str()), "XGBoosterSetParam" );
	}
	
	Booster::Booster( const std::vector<char>& model )noexcept(false):
		TrainRowCount{0},
		ValidationRowCount{0}
	{
		LoadModelFromBinary( model );
		var featureNames = GetAttribute( "feature_names" );
		if( featureNames.size() )
			_featureNames = StringUtilities::Split( featureNames );
	}

	void Booster::Save( const fs::path& path )noexcept(false)
	{
		auto dumpPath = path; dumpPath.replace_extension( ".dump" );
		DBG( "Saving model:  '{}'/'{}'.", path.string(), dumpPath.string() );
		//var& features = _pTrain->Features();
		SetAttribute( "feature_names", StringUtilities::AddCommas(FeatureNames()) );
		CALL( XGBoosterSaveModel(_handle, path.string().c_str()), "XGBoosterSaveModel" );

		var pTrees = Booster::StringTrees( true );
		ofstream os( dumpPath.string() );
		for( var& tree : *pTrees )
			os << tree;
	}
	sp<const Dataset> Booster::ValidationPtr()const noexcept
	{
		return _pValidation;
	}

	void Booster::LoadModelFromBinary( const vector<char>& bytes )noexcept(false)
	{
		Release();
		CALL( XGBoosterCreate(nullptr, 0, &_handle), "XGBoosterCreate" );
		CALL( XGBoosterLoadModelFromBuffer(_handle, bytes.data(), bytes.size()), "XGBoosterLoadModelFromBuffer" );
 	}

	MapPtr<string,double> Booster::FeatureImportances( EFeatureImportance featureImportance )const noexcept(false)
	{
		//'weight', 'gain', 'cover', 'total_gain', 'total_cover'
		auto pImportances = make_shared<map<string,double>>();
		var weight = featureImportance==EFeatureImportance::Weight;
		var pTrees = StringTrees( !weight );
		
		function<void(const TreeNode& node, function<void(const TreeNode& node)>& todo)> nodeRecursion = [&nodeRecursion]( const TreeNode& node, function<void(const TreeNode& node)>& todo )
		{
			ASSERT( node.FeatureName.size() );
			todo( node );
			if( node.YesNodePtr && !node.YesNodePtr->IsLeaf() )
				nodeRecursion( *node.YesNodePtr, todo );
			if( node.NoNodePtr && !node.NoNodePtr->IsLeaf() )
				nodeRecursion( *node.NoNodePtr, todo );
		};
		auto calculate = [&pTrees, &nodeRecursion]( function<void(const TreeNode& node)>& todo )
		{
			for( var& tree : *pTrees )
			{
				TreeNode node{ nlohmann::json::parse(tree) };
				if( !node.IsLeaf() )
					nodeRecursion( node, todo );
			}
		};
		if( weight )
		{
			function<void(const TreeNode& node)> calc = [&pImportances]( const TreeNode& node )
			{
				auto iteratorInserted = pImportances->emplace( node.FeatureName, 1 );
				if( !iteratorInserted.second )
					iteratorInserted.first->second+=1;
			};
			calculate( calc );
		}
		else
		{
			var isGain = (featureImportance & EFeatureImportance::Gain)==EFeatureImportance::Gain;
			map<string,uint> fmap;
			function<void(const TreeNode& node)> calc = [&pImportances, &fmap, &isGain]( const TreeNode& node )
			{
				auto iteratorInserted = fmap.emplace( node.FeatureName, 1 );
				double importance = isGain ? node.Gain : static_cast<double>(node.Cover);
				if( iteratorInserted.second )
					pImportances->emplace( node.FeatureName, importance );
				else
				{
					iteratorInserted.first->second += 1;
					(*pImportances)[node.FeatureName] += importance;
				}
			};
			calculate( calc );
			if( (featureImportance & EFeatureImportance::Total)==EFeatureImportance::None )
			{
				for( auto& [feature, value] : *pImportances )
					value = value/fmap[feature];
			}
		}
		return pImportances;
	}

	string Booster::GetAttribute( const string& name )const noexcept(false)
	{
		const char* pszResult;
		int success;
		CALL( XGBoosterGetAttr(_handle, name.c_str(), &pszResult, &success), "XGBoosterGetAttr" );
		return success ? string(pszResult) : string();
	}
	void Booster::SetAttribute( const string& name, const string& value )noexcept(false)
	{
		CALL( XGBoosterSetAttr(_handle, name.c_str(), value.c_str()), "XGBoosterGetAttr" );
	}

	uint Booster::FeatureCount()const noexcept(false)
	{
		return FeatureNames().size();
	}

	const vector<string>& Booster::FeatureNames()const noexcept
	{
		ASSERT( _featureNames.size() || _pTrain );
		return _featureNames.size() ? _featureNames : _pTrain->Features();
	}
	
	Booster::~Booster()
	{
		Release();
	}
	void Booster::Release()noexcept
	{
		if( _handle )
		{
			var failed = XGBoosterFree( _handle );
			_handle = nullptr;
			if( failed )
				ERR( "({}) - {}", failed, XGBGetLastError() );
		}
	}
	void Booster::SetBestIteration( uint value )noexcept 
	{
		IBooster::SetBestIteration( value ); 
		bst_ulong out_len; const char *out_dptr;
		CALL( XGBoosterGetModelRaw(_handle, &out_len, &out_dptr), "XGBoosterGetModelRaw" );
		_bestIterationValue.clear();_bestIterationValue.reserve( out_len );
		_bestIterationValue.insert( _bestIterationValue.end(), out_dptr, out_dptr+out_len );

	}
	void Booster::LoadBestIteration()noexcept(false)
	{
		ASSERT( BestIteration() && _bestIterationValue.size() );
		CALL( XGBoosterLoadModelFromBuffer( _handle, _bestIterationValue.data(), _bestIterationValue.size()), "XGBoosterLoadModelFromBuffer" );
	}
	
	VectorPtr<string> Booster::StringTrees( bool withStats )const noexcept(false)
	{
		var& features = FeatureNames();
		var featureSize = features.size();  ASSERT( featureSize );
		vector<const char*> featureCharPtrs; featureCharPtrs.reserve( featureSize );
		vector<const char*> types; types.reserve( featureSize );
		const char* pszType = "float";
		for( var& feature : features )
		{
		//	DBG( "pushing back '{}'", feature.c_str() );
			featureCharPtrs.push_back( feature.c_str() );
			types.push_back( pszType );
		}
		constexpr const char* pszFormat = "json";//json or not json
		bst_ulong length;
		const char** ppszResults;
		CALL( XGBoosterDumpModelExWithFeatures(_handle, static_cast<int>(featureSize), featureCharPtrs.data(), types.data(), (withStats ? 1 : 0), pszFormat, &length, &ppszResults ), "XGBoosterDumpModelWithFeatures" );
		auto pResults = make_shared<vector<string>>(); pResults->reserve( length );
		for( uint i=0; i<length; ++i )
			pResults->push_back( ppszResults[i] );
		return pResults;
	}

	string Booster::to_string( uint iterationNumber )const noexcept(false)
	{
		var pStrings = StringTrees( true );
		ostringstream os;
		for( var string: *pStrings )
			os << string << std::endl;
		return os.str();
	}

	//you can train on mutiple objectives, aoc, rmse, etc.  #of objectives
	uint Booster::GetEvaluationCounts()const noexcept(false)
	{
		return 1;
	}
	

	vector<double> Booster::GetEvaluation( bool validation, uint iteration )const noexcept(false)
	{
		ASSRT_TR( !validation || _pValidation );
		auto handle = validation ? ValidationPtr()->Handle() : _handle;
		const char* pszNames = validation ? "validation" : "train";
		const char* pszResult;
		CALL( XGBoosterEvalOneIter(_handle, (int)iteration, &handle, &pszNames, 1, &pszResult), "XGBoosterEvalOneIter" );
		var columns = StringUtilities::Split( string(pszResult), ':' );
		ASSRT_TR( columns.size()==2 );
		vector<double> results; results.reserve( GetEvaluationCounts() );
		results.push_back( stof(columns.back()) );
//		ASSRT_EQ( results.size(), (uint)outLength );

		return results;
	}
	
	vector<float> Booster::Predict( const Dataset& ds )
	{
		bst_ulong out_len;
		const float* pResults;
		CALL( XGBoosterPredict( _handle, ds.Handle(), 0/*option_mask=normal*/, 0 /*ntree_limit*/, &out_len, &pResults), "XGBoosterPredict" );
		ASSERT( out_len>0 );
		return vector<float>{ pResults, pResults+out_len };
	}
	sp<vector<double>> Booster::Predict( const XgbMatrix& matrix )noexcept(false)
	{
		Dataset ds{matrix};
		var results = Predict( ds );
		//const char* pszParameter = "";
		return make_shared<vector<double>>( results.begin(), results.end() );
	}
	double Booster::Predict( const Math::RowVector<float,-1>& vector )noexcept(false)
	{
		Dataset ds{ vector };
		return static_cast<double>( Predict(ds).front() );
	}
	double Booster::Predict( const double* pFeatures )noexcept(false)
	{
		var size = ColumnCount();
		vector<float> floatValues; floatValues.reserve( size );
		for( uint i=0; i<size; ++i )
			floatValues.push_back( static_cast<float>(*(pFeatures+i)) );
		Dataset ds{ floatValues.data(), 1, size };
		return static_cast<double>( Predict(ds).front() );
	}
	bool Booster::UpdateOneIteration( int index )noexcept(false)
	{
		CALL( XGBoosterUpdateOneIter(_handle, index, _pTrain->Handle()), "XGBoosterUpdateOneIter" );
		CALL( XGBoosterSaveRabitCheckpoint(_handle), "XGBoosterSaveRabitCheckpoint" );
		return true;
	}

/*	uint Booster::SaveIfElse( const fs::path& modelPath, string_view namespaceName, ostream& osCpp )noexcept(false)
	{
		LightGBM::Boosting* pBoosting;
		try
		{
			pBoosting = LightGBM::Boosting::CreateBoosting( "gbdt", modelPath.string().c_str() );
		}
		catch( const std::runtime_error& e )
		{
			THROW( IOException( "Could not parse {} - ", modelPath.string(), e.what() ) );
		}
		pBoosting->MyModelToIfElse( string(namespaceName), osCpp );
		return pBoosting->MaxFeatureIdx()+1;
	}
	uint Booster::ModelCount( const fs::path& modelPath )noexcept(false)
	{
		//LightGBM::Boosting* pBoosting;
		uint count = 0;
		try
		{
			auto pBoosting = LightGBM::Boosting::CreateBoosting("gbdt", modelPath.string().c_str() );
			count = pBoosting->NumberOfTotalModel();
		}
		catch( const std::runtime_error& e )
		{
			THROW( IOException( "Could not parse {} - ", modelPath.string(), e.what() ) );
		}
		return count;
	}*/

}