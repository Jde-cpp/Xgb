#include "stdafx.h"
#include "XgbBoosterParams.h"
#define var const auto

namespace Jde::AI::Dts::Xgb
{
#pragma region Constructors
	XgbBoosterParams::XgbBoosterParams()noexcept:
		XgbBoosterParams( "reg:linear"sv )
	{}

	XgbBoosterParams::XgbBoosterParams( string_view objective )noexcept:
		Dts::IBoosterParams{ XgbDoubleParams, XgbStringParams, XgbUIntParams }
	{
		if( !XgbDoubleParams.size() )
		{
			std::for_each( IBoosterParams::DefaultDoubleParams.begin(), IBoosterParams::DefaultDoubleParams.end(), [&](var& value){ XgbDoubleParams.emplace(value); } );
			std::for_each( IBoosterParams::DefaultStringParams.begin(), IBoosterParams::DefaultStringParams.end(), [&](var& value){ XgbStringParams.emplace(value); } );
			std::for_each( IBoosterParams::DefaultUIntParams.begin(), IBoosterParams::DefaultUIntParams.end(), [&](var& value){ XgbUIntParams.emplace(value); } );

			XgbDoubleParams.emplace( Gamma.Name );
			XgbDoubleParams.emplace( MinChildWeight.Name );
			XgbDoubleParams.emplace( MaxDeltaStep.Name );
			XgbDoubleParams.emplace( SubSample.Name );
			XgbDoubleParams.emplace( ColSampleByTree.Name );
			XgbDoubleParams.emplace( ColSampleByLevel.Name );
			XgbDoubleParams.emplace( ColSampleByNode.Name );
			XgbDoubleParams.emplace( Lambda.Name );
			XgbDoubleParams.emplace( Alpha.Name );
			XgbDoubleParams.emplace( ScalePosWeight.Name );
			
			XgbUIntParams.emplace( "nthread" );
			XgbUIntParams.emplace( "verbosity" );
			XgbUIntParams.emplace( "seed" );
			XgbUIntParams.emplace( "silent" );
			XgbStringParams.emplace( "predictor" );
			XgbStringParams.emplace( "eval_metric" );
		}
		_parameters.emplace( Gamma.Name, make_shared<TParameter<double>>(Gamma) );
		_parameters.emplace( MinChildWeight.Name, make_shared<TParameter<double>>(MinChildWeight) );
		_parameters.emplace( MaxDeltaStep.Name, make_shared<TParameter<double>>(MaxDeltaStep) );
		_parameters.emplace( SubSample.Name, make_shared<TParameter<double>>(SubSample) );
		_parameters.emplace( ColSampleByTree.Name, make_shared<TParameter<double>>(ColSampleByTree) );
		_parameters.emplace( ColSampleByLevel.Name, make_shared<TParameter<double>>(ColSampleByLevel) );
		_parameters.emplace( ColSampleByNode.Name, make_shared<TParameter<double>>(ColSampleByNode) );
		_parameters.emplace( Lambda.Name, make_shared<TParameter<double>>(Lambda) );
		_parameters.emplace( Alpha.Name, make_shared<TParameter<double>>(Alpha) );
		_parameters.emplace( ScalePosWeight.Name, make_shared<TParameter<double>>(ScalePosWeight) );
		_parameters.emplace( Predictor.Name, make_shared<TParameter<string>>(Predictor) );
		_parameters.emplace( Verbosity.Name, make_shared<TParameter<uint>>(Verbosity) );
		_parameters.emplace( Seed.Name, make_shared<TParameter<uint>>(Seed) );
		_parameters.emplace( Silent.Name, make_shared<TParameter<uint>>(Silent) );
		auto pObjective = make_shared<TParameter<string>>( Objective );
		pObjective->Initial = objective;
		_parameters.emplace( Objective.Name, pObjective );

		//try to use default
		//auto pObjective = make_shared<TParameter<string>>( EvalMetric );
		//if( objective=="reg:linear" || objective=="reg:linear" )
		//_parameters.emplace( EvalMetric.Name, make_shared<TParameter<string>>(EvalMetric) );

		auto pThread = make_shared<TParameter<uint>>( ThreadCount ); pThread->Name = "nthread";
		_parameters.emplace( pThread->Name, pThread );
	}

	XgbBoosterParams::XgbBoosterParams( const fs::path& paramFile )noexcept(false):
		XgbBoosterParams()
	{
		if( !fs::exists(paramFile) )
			THROW( LogicException( fmt::format("{} does not exist.", paramFile.string())) );

		std::ifstream is( paramFile );
		//is.exceptions( std::ios::failbit | std::ios::badbit );
		//is.open( filename.c_str() );
		if( !is.good() )
			THROW( LogicException( fmt::format("{} could not read.", paramFile.string())) );
		Read( is );
	}
	XgbBoosterParams::XgbBoosterParams( std::istream& is )noexcept:
		XgbBoosterParams()
	{
		Read( is );
	}
	nlohmann::json ToJson( const XgbBoosterParams& params ) noexcept
	{
		nlohmann::json j;
		j["best_iteration"] = params.BestIteration();
		j["parameters"] = nlohmann::json::array();
		for( var& nameParameter : params.Parameters() )
			j["parameters"].push_back( nameParameter.second->ToJson() );
		return j;
	}
	std::ostream& operator<<( std::ostream& os, const XgbBoosterParams& parameters )noexcept
	{
		os << ToJson(parameters) << endl;
		return os;
	}
#pragma endregion
#pragma region XgbBoosterParams
	uint XgbBoosterParams::NumberOfLeavesValue()const noexcept
	{
		var pParam = std::dynamic_pointer_cast<TParameter<uint>>( (*this)["num_leaves"] );
		return pParam->Initial;
	}
	string XgbBoosterParams::GetMetric()const noexcept
	{
		string metric;
		string key = _parameters.find("eval_metric")==_parameters.end() ? Objective.Name : string("eval_metric");
		var pMetric = std::dynamic_pointer_cast<TParameter<string>>( (*this)[key] );
		return pMetric->Initial;
	}
	void XgbBoosterParams::SetMetric( string_view metric )noexcept
	{
		if( !metric.size() )
			metric = "reg:linear";
		if( _parameters.find("eval_metric")==_parameters.end() )
		{
			auto objective = metric;
			if( metric=="reg:linear" )
				objective = "rmse";
			_parameters["eval_metric"] = make_shared<TParameter<string>>( "eval_metric", string(objective) );
		}
/*		else if( metric=="l1" )
			objective = "regression_l1";
		else if( metric=="quantile" || metric=="huber" || metric=="fair" || metric=="poisson" || metric=="gamma" || metric=="mape" || metric=="tweedie" )
			objective = metric;
		else
		{
			objective = metric;  WARN( "Unknown metric '{}'", metric );
		}*/
		//auto pObjective = make_shared<TParameter<string>>( "eval_metric", objective );
		//pObjective->Initial = "reg:linear";
		//_parameters.emplace( Objective.Name, pObjective );
		// auto nameParamInserted = _parameters.emplace( "eval_metric",  );
		// if( !nameParamInserted.second )
		// {
		// 	nameParamInserted.first->second->Initial = objective;
		// }
	}

/*	void XgbBoosterParams::FixObjective(TParameter<string>& parameter)const noexcept
	{
		if( parameter.Initial=="regression" )
			parameter.Initial = "reg:linear";
	}
*/
	string XgbBoosterParams::DeviceValue()const noexcept
	{
		var pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["predictor"] );
		return pDevice->Initial=="cpu_predictor" ? string("cpu") : string("gpu");
	}
	void XgbBoosterParams::SetCpu()const noexcept
	{
		auto pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["predictor"] );
		pDevice->Initial = "cpu_predictor";
	}
	void XgbBoosterParams::SetGpu()const noexcept
	{
		auto pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["predictor"] );
		pDevice->Initial = "gpu_predictor";
	}

	const Parameter* XgbBoosterParams::FindDefault( string_view name )const noexcept
	{
		auto pParameter = IBoosterParams::FindDefault( name );
		if( !pParameter )
		{
			if( name==Gamma.Name )
				pParameter = &Gamma;
			else if( name==MinChildWeight.Name )
				pParameter = &MinChildWeight;
			else if( name==MaxDeltaStep.Name )
				pParameter = &MaxDeltaStep;
			else if( name==SubSample.Name )
				pParameter = &SubSample;
			else if( name==ColSampleByTree.Name )
				pParameter = &ColSampleByTree;
			else if( name==ColSampleByLevel.Name )
				pParameter = &ColSampleByLevel;
			else if( name==ColSampleByNode.Name )
				pParameter = &ColSampleByNode;
			else if( name==Lambda.Name )
				pParameter = &Lambda;
			else if( name==Alpha.Name )
				pParameter = &Alpha;
			else if( name==ScalePosWeight.Name )
				pParameter = &ScalePosWeight;
			else if( name==Verbosity.Name )
				pParameter = &Verbosity;
			else if( name==Predictor.Name )
				pParameter = &Predictor;
			else if( name==EvalMetric.Name )
				pParameter = &EvalMetric;
			else if( name==Seed.Name )
				pParameter = &Seed;
			else if( name==Silent.Name )
				pParameter = &Silent;
			else
				THROW( LogicException("Could not find parameter '{}'.", name) );
		}
		return pParameter;
	}
	//TParameter( string_view name, const T& initial, const T& delta, T minimum, T maximum, bool hasRange=true );
	const TParameter<double> XgbBoosterParams::Gamma = TParameter<double>( "gamma", 0, 1 );
	const TParameter<double> XgbBoosterParams::MinChildWeight = TParameter<double>( "min_child_weight", 1, 1 );
	const TParameter<double> XgbBoosterParams::MaxDeltaStep = TParameter<double>( "max_delta_step", 0, 1 );
	const TParameter<double> XgbBoosterParams::SubSample  = TParameter<double>( "subsample", 1, .05, .001, 1 );
	const TParameter<double> XgbBoosterParams::ColSampleByTree = TParameter<double>( "colsample_bytree",  1, .1, .001, 1 );
	const TParameter<double> XgbBoosterParams::ColSampleByLevel = TParameter<double>( "colsample_bylevel",1, .05, .001, 1 );
	const TParameter<double> XgbBoosterParams::ColSampleByNode  = TParameter<double>( "colsample_bynode", 1, .05, .001, 1 );
	const TParameter<double> XgbBoosterParams::Lambda = TParameter<double>( "lambda", 1, .05, 0, 1.5 );
	const TParameter<double> XgbBoosterParams::Alpha = TParameter<double>( "alpha", 0, .1, 0, 1.5 );
	const TParameter<double> XgbBoosterParams::ScalePosWeight = TParameter<double>( "scale_pos_weight", 1, .05, 0, 1.5 );
	const TParameter<uint> XgbBoosterParams::Verbosity = TParameter<uint>( "verbosity", 1 );//0 (silent), 1 (warning), 2 (info), 3 (debug). 
	const TParameter<string> XgbBoosterParams::Predictor = TParameter<string>( "predictor", "cpu_predictor" );
	const TParameter<string> XgbBoosterParams::EvalMetric = TParameter<string>( "eval_metric", "rmse" );
	const TParameter<uint> XgbBoosterParams::Seed = TParameter<uint>( "seed", 99 );//0 (silent), 1 (warning), 2 (info), 3 (debug). 
	const TParameter<uint> XgbBoosterParams::Silent = TParameter<uint>( "silent", 1 );//0 (silent), 1 (warning), 2 (info), 3 (debug). 

	set<string> XgbBoosterParams::XgbDoubleParams;
	set<string> XgbBoosterParams::XgbStringParams;
	set<string> XgbBoosterParams::XgbUIntParams;
#pragma endregion
}