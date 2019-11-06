#pragma once
#include <sstream>
#include "../../DecisionTree/source/IBoosterParams.h"
#include "Exports.h"

namespace Jde::AI::Dts::Xgb
{
#pragma region BoosterParams
	struct JDE_XGB_VISIBILITY XgbBoosterParams : public Dts::IBoosterParams
	{
		IBoosterParamsPtr Create()const noexcept override{ return make_shared<XgbBoosterParams>(); }
		IBoosterParamsPtr Clone()const noexcept override{ return make_shared<XgbBoosterParams>(*this); }

		XgbBoosterParams()noexcept;
		XgbBoosterParams( string_view objective )noexcept;
		XgbBoosterParams( std::istream& is )noexcept;
		XgbBoosterParams( const fs::path& path )noexcept(false);
		XgbBoosterParams( const XgbBoosterParams& )=default;
		//void Initialize();
		uint NumberOfLeavesValue()const noexcept;
		string GetMetric()const noexcept;void SetMetric( string_view metric )noexcept;

		string DeviceValue()const noexcept override; void SetCpu()const noexcept  override; void SetGpu()const noexcept  override;/*const because not significant*/
		string_view ThreadParamName()const noexcept override{return "nthread"sv;}
		const Parameter* FindDefault( string_view name )const noexcept override;

		static set<string> XgbDoubleParams;
		static set<string> XgbStringParams;
		static set<string> XgbUIntParams;
		const static TParameter<double> Gamma;//min_split_loss, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.  range: [0,∞]
		const static TParameter<double> MinChildWeight;
		const static TParameter<double> MaxDeltaStep;
		const static TParameter<double> SubSample;
		const static TParameter<double> ColSampleByTree;
		const static TParameter<double> ColSampleByLevel;
		const static TParameter<double> ColSampleByNode;
		const static TParameter<double> Lambda;
		const static TParameter<double> Alpha;
		const static TParameter<double> ScalePosWeight;
		const static TParameter<uint> Verbosity;
		const static TParameter<string> Predictor;
		const static TParameter<string> EvalMetric;
		const static TParameter<uint> Seed;
		const static TParameter<uint> Silent;

		//const static TParameter<string> Verbose;
	};
	std::ostream& operator<<( std::ostream& os, const XgbBoosterParams& parameter )noexcept;
#pragma endregion
}