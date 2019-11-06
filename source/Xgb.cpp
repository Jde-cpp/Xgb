#pragma region Includes
#include "stdafx.h"
#include "Xgb.h"
#include <sstream>
#include <list>
#include "../../Framework/source/io/File.h"
#include "XgbBoosterParams.h"
#include "Booster.h"
#include "Dataset.h"

#define var const auto
#pragma endregion Includes

Jde::AI::Dts::IDecisionTree* GetDecisionTree()
{
	return new Jde::AI::Dts::Xgb::DecisionTree();
}

namespace Jde::AI::Dts::Xgb
{
#pragma region Defines
	using Eigen::MatrixXf;
	using Eigen::VectorXf;
	constexpr uint TrainingRounds = 1000;
#pragma endregion
	sp<IBooster> DecisionTree::CreateBooster( const fs::path& file )const noexcept(false)
	{
		std::unique_ptr<std::vector<char>> pContents = IO::FileUtilities::LoadBinary( file );
		return make_shared<Booster>( *pContents );
	}
	sp<IBooster> DecisionTree::CreateBooster( const IBoosterParams& params, sp<const IDataset>& pTrain, sp<const IDataset> pValidation )
	{
		return make_shared<Booster>( params, pTrain, pValidation );
	}
	// fs::path DecisionTree::BaseDir()const noexcept
	// {
	// 	return Settings::Global().Get<fs::path>( "DtsBaseDir", fs::path{"/mnt/2TB/dts"} );
	// }
	IBoosterParamsPtr DecisionTree::LoadParams( const fs::path& file )const noexcept(false)
	{
		ifstream is(file);
		if( !is.good() )
			THROW( Exception( fmt::format("Could not open file {}", file.string()) ) );
		return make_shared<AI::Dts::Xgb::XgbBoosterParams>( is );
		//return make_shared<AI::Dts::Xgb::XgbBoosterParams>();
	}
	IBoosterParamsPtr DecisionTree::LoadDefaultParams( string_view objective )const noexcept(false)
	{
		return make_shared<AI::Dts::Xgb::XgbBoosterParams>( objective );
	}
	sp<IDataset> DecisionTree::CreateDataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames/*=nullptr*/, shared_ptr<const IDataset> pTrainingDataset )
	{
		return make_shared<Dataset>( matrix, pColumnNames, y );// ? pTrainingDataset.get() : nullptr
	}

	void XgbLog( const char* pszValue )
	{
		DBG( "Xgb:  {}", pszValue );
	}

	void DecisionTree::RegisterLogCallback()
	{
		XGBRegisterLogCallback( XgbLog );
	}
	

//#pragma region Defines
//	const fs::path DTBaseDir{"/mnt/2TB/LightGbm"};
//	using Eigen::MatrixXf;
//	using Eigen::VectorXf;
//	constexpr uint TrainingRounds = 1000;
	/*
	sp<AI::IBoosterParams> LoadParams2( const fs::path& file )noexcept(false)
	{
		ifstream is(file);
		if( !is.good() )
			THROW( Exception( fmt::format("Could not open file {}", file.string()) ) );
		return make_shared<AI::Xgb::BoosterParams>( is );
	}
	*/
/*	sp<AI::IBoosterParams> DecisionTree::LoadParams( uint featureLength )noexcept(false)
	{
		var pDirFiles = IO::FileUtilities::GetDirectory( DTBaseDir );
		static map<uint,fs::path> files;
		if( !files.size() )
		{
			for( var& file : *pDirFiles )
			{
				var stem = file.path().stem().string();
				if( file.path().extension()==".params" && StringUtilities::StartsWith(stem, "lgb") )
				{
					uint featureCount = StringUtilities::TryTo<uint>( stem.substr(3), 0 );
					if( featureCount!=0 )
						files.emplace( featureCount, file.path() );
				}
			}
		}
		if( !files.size() )
			THROW( Exception("!files.size()") );
		auto fileName = DTBaseDir/"tune.params";
		if( featureLength<81 )
		{
			auto pLowerBound = files.lower_bound( featureLength );
			if( pLowerBound!=files.begin() && (pLowerBound==files.end() || pLowerBound->first!=featureLength) )
				--pLowerBound;
			fileName = pLowerBound->second;
		}
		return LoadParams2( fileName );
	}
	*/
//	shared_ptr<Jde::AI::Dts::IBooster> Train2( const Dataset& train, const IBoosterParams& params, uint count, const Dataset* pValidation=nullptr, uint earlyStoppingRounds=std::numeric_limits<uint>::max() )noexcept(false);
	
//#pragma endregion
	/*
	shared_ptr<Jde::AI::Dts::IBooster> Train( const Eigen::MatrixXf& x, const Eigen::VectorXf& y, const IBoosterParams& params, uint count, const std::vector<string>& columnNames )noexcept(false)
	{
		return Train2( Dataset(x, y, &params, &columnNames, nullptr), params, count );
	}
	*/
	/*
	map<BoosterParams,ParamResults> ReadPrevious( const fs::path& csvFile )
	{
		std::ifstream is( csvFile );
		ASSRT_TR( is.good() );
		string line;
		getline( is, line );
		var columnNameList = StringUtilities::Split( line );
		var columnNames = vector<string>( columnNameList.begin(), columnNameList.end() );
		map<BoosterParams,ParamResults> previousExecutions;
		while( is.good() )
		{
			getline( is, line );
			if( line.size()==0 )
				continue;
			var valueList = StringUtilities::Split( line );
			var values = vector<string>( valueList.begin(), valueList.end() );
			ASSRT_EQ( columnNames.size(), values.size() );
			BoosterParams paramsOther;
			for( uint i=5; i<values.size(); ++i )
			{
				var& columnName = columnNames[i];
				shared_ptr<Parameter> pParameter = paramsOther[columnName];
				if( pParameter->Type()==ParameterType::String )
					dynamic_pointer_cast<TParameter<string>>( pParameter )->Initial = values[i];
				else if( pParameter->Type()==ParameterType::Double )
					dynamic_pointer_cast<TParameter<double>>( pParameter )->Initial = stod( values[i] );
				else if( pParameter->Type()==ParameterType::UInt )
					dynamic_pointer_cast<TParameter<uint>>( pParameter )->Initial = stoul( values[i] );
			}
			paramsOther.BestIteration = stoul( values[4] );
			var results = ParamResults( paramsOther,stod(values[0]), stod(values[1]), stod(values[2]), stod(values[3]), paramsOther.BestIteration );
			var inserted = previousExecutions.emplace(paramsOther, results ).second;
			if( !inserted )
				GetDefaultLogger()->critical( "Duplicate parameter sets {}", paramsOther.ToString() ); //ASSRT_TR( inserted );
		}
		return previousExecutions;
	}*/
/*	tuple<BoosterParams,bool> TuneParam( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, const fs::path& saveStem, uint foldCount, BoosterParams& currentRun, map<BoosterParams,ParamResults>& previousExecutions, string_view logRemainder )noexcept;
	BoosterParams Tune( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, uint testCount, const fs::path& saveStem, uint foldCount, const fs::path& paramStart )noexcept(false)
	{
		BoosterParams bestParams;
		auto paramFile = fs::path(saveStem).replace_extension(".params");
		if( fs::exists(paramFile) )
			bestParams = BoosterParams( paramFile );
		else if( fs::exists(paramStart) )
			bestParams = BoosterParams( paramStart );
		DBG( "BaggingFractionValue={}", bestParams.BaggingFractionValue() );
		bestParams.SetGpu();
		auto csvFile = fs::path(saveStem).replace_extension( ".csv" );
		auto previousExecutions = fs::exists(csvFile) ? ReadPrevious(csvFile) : map<BoosterParams,ParamResults>();

		for( uint roundIndex=0; roundIndex<testCount; ++roundIndex )
		{
			uint parameterIndex = 0;
			var hasRangeCount = bestParams.HasRangeCount();
			for( var& changing : bestParams.Parameters() )
			{
				var& parameter = *changing.second;
				if( !parameter.HasRange )
					continue;
				if( parameter.Name=="min_sum_hessian_in_leaf" )
				{
					CRITICAL( "Trying to change min_sum_hessian_in_leaf {}", "none" );
					continue;
				}
				auto currentRun = bestParams;
				currentRun[parameter.Name]->SetRange();
				DBG( "BaggingFractionValue={}", currentRun.BaggingFractionValue() );
				var logSuffix = fmt::format( "({}/{})({}/{})", ++parameterIndex, hasRangeCount, roundIndex, testCount );
				bestParams = get<0>( TuneParam(xs, ys, saveStem, foldCount, currentRun, previousExecutions, logSuffix) );
			}
		}
	 	return bestParams;
	}
	*/
/*	BoosterParams TuneOne( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, const fs::path& saveStem, uint foldCount, string_view parameterName )noexcept(false)
	{
		var paramFile = fs::path( saveStem ).replace_extension( ".params" );
		BoosterParams bestParams( paramFile );
		bestParams.SetGpu();
		var csvFile = fs::path(saveStem).replace_extension( ".csv" );
		if( !fs::exists(csvFile) )
			THROW( LogicException(fmt::format("{} was not found", csvFile))  );

		map<BoosterParams,ParamResults> previousExecutions = ReadPrevious( csvFile );
		for(uint roundIndex=0;; ++roundIndex)
		{
			auto currentRun = bestParams;
			currentRun[parameterName]->SetRange();
			var logSuffix = fmt::format( "({}/{}) - {}", 0, 1, roundIndex );
			var results = TuneParam( xs, ys, saveStem, foldCount, currentRun, previousExecutions, logSuffix );
			bestParams = get<0>( results );
			if( !get<1>( results) )
				break;
		}
		return bestParams;
	}

	tuple<BoosterParams,bool> TuneParam( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, const fs::path& saveStem, uint foldCount, BoosterParams& currentRun, map<BoosterParams,ParamResults>& previousExecutions, string_view logRemainder )noexcept
	{
		auto parameterSets = currentRun.GetSets();
		uint parameterSetIndex=0;
		uint executionCount = 0;
		for( var& parameterSet : parameterSets )
		{
			if( previousExecutions.find(parameterSet)==previousExecutions.end() )
				++executionCount;
		}
		if( executionCount==0 )
			return make_tuple( currentRun, false );

		var poolThreadCount = std::min<uint>( executionCount, std::max<uint>(std::thread::hardware_concurrency()-2,1) );
		{
			Threading::Pool pool( poolThreadCount );
			std::mutex previousExecutionsMutex;
			for( var& parameterSet : parameterSets )
			{
				if( previousExecutions.find(parameterSet)!=previousExecutions.end() )
					continue;
				/ *if( parameterSet.NumberOfLeavesValue()<2 )
				{
					GetDefaultLogger()->critical( "parameterSet.NumberOfLeavesValue()<2" );	
					continue;
				}* /
				ASSRT_TR( previousExecutions.find(parameterSet)==previousExecutions.end() );
				DBG0( parameterSet.ToString() );
				var logSuffix = fmt::format( "({{}}/{})({}/{}){}", xs.size(), parameterSetIndex++, parameterSets.size(), logRemainder );
				//var logSuffix = fmt::format( "({{}}/{})({}/{})({}/{})({}/{})", xs.size(), parameterSetIndex++, parameterSets.size(), parameterIndex, hasRangeCount, roundIndex, testCount );
				auto func = [&previousExecutions, &previousExecutionsMutex, &ys, &xs, logSuffix, &parameterSet, &foldCount, poolThreadCount]()
				{
					uint groupIndex = 0;
					map<uint,vector<double>> groupFolds;
					uint maxIteration = 0;
					vector<Math::VPtr<>>::const_iterator ppY=ys.begin();
					for( vector<unique_ptr<Eigen::MatrixXf>>::const_iterator ppX=xs.begin(); ppX!=xs.end() && ppY!=ys.end(); ++ppX, ++ppY )
					{
						var logSuffix2 = fmt::format( logSuffix, groupIndex );
						parameterSet.SetThreadCount( poolThreadCount==1 ? std::thread::hardware_concurrency()-2 : 1 );
						tuple<uint,vector<double>> result = CrossValidate( parameterSet, *(*ppX), *(*ppY), foldCount, TrainingRounds, logSuffix2 );
						maxIteration = std::max( get<0>(result), maxIteration );
						groupFolds[groupIndex++] = get<1>( result );
					}
					std::unique_lock l( previousExecutionsMutex );
					previousExecutions.emplace( parameterSet, ParamResults(parameterSet, maxIteration, groupFolds) );
				};
				pool.Submit( func );	
			}
		}
		ASSRT_GT( 0, previousExecutions.size() );

		var bestParams = SaveTuning( saveStem, previousExecutions );
		return make_tuple( bestParams, true );
	}

#pragma region SaveTuning
	BoosterParams SaveTuning( const fs::path& saveStem, const map<BoosterParams,ParamResults>& previousExecutions )noexcept(false)
	{
		ASSRT_GT( (uint)0, previousExecutions.size() );
		multimap<ParamResults,BoosterParams> sorted;
		for( var& [params, results] : previousExecutions )
		{
			params.BestIteration = results.BestIteration;
			sorted.emplace( results, params );
		}
		var pTop = sorted.begin();
		{
			auto paramPath = saveStem; 
			std::ofstream os;
			os.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			os.open( paramPath.replace_extension(".params") );
			//if( os.fail() )
				//THROW( CodeException(fmt::format("Could not write to {}", paramPath.c_str()), errno) );
			os << pTop->second;
		}
		auto resultsPath = saveStem;
		std::ofstream os;
		os.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		os.open( resultsPath.replace_extension(".csv") );
		//if( os.fail() )
			//THROW( CodeException(fmt::format("Could not write to {}", resultsPath.c_str()), errno) );
		os << "average,std_dev,min,max,iteration";
		for( var& parameter : pTop->second.Parameters() )
			os <<","<< parameter.first;
		os << std::endl;
		for( var& [results, params] : sorted )
		{
			os << results.Average() << "," << results.StdDeviation() << "," << results.Min() << "," << results.Max() << "," << results.BestIteration;
			for( var& parameter : params.Parameters() )
				os << "," << parameter.second->InitialString();
			os << std::endl;
		}
		return pTop->second;
	}
		*/
#pragma endregion
}	