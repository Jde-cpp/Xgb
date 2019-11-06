
#include "Exports.h"
#include "../../DecisionTree/source/IDecisionTree.h"

extern "C" JDE_XGB_VISIBILITY Jde::AI::Dts::IDecisionTree* GetDecisionTree(); 
namespace Jde::AI::Dts::Xgb
{
	using std::vector;
	struct XgbBoosterParams;
	class Booster;
	struct JDE_XGB_VISIBILITY DecisionTree : public IDecisionTree
	{
		DecisionTree():IDecisionTree{"xgb"}{}
		sp<IBooster> CreateBooster( const fs::path& file )const noexcept(false)override;
		sp<IBooster> CreateBooster( const IBoosterParams& params, sp<const IDataset>& train, sp<const IDataset> pValidation )override;
		//fs::path BaseDir()const noexcept override;
		IBoosterParamsPtr LoadParams( const fs::path& file )const noexcept(false) override;
		IBoosterParamsPtr LoadDefaultParams( string_view objective )const noexcept(false) override;
		sp<IDataset> CreateDataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames/*=nullptr*/, shared_ptr<const IDataset> pTrainingDataset )override;
		string_view DefaultRegression()const noexcept override{ return "reg:linear"sv; }
		static void RegisterLogCallback();
	};
	

	/*
	JDE_XGB_VISIBILITY sp<IBooster> Train( const Eigen::MatrixXf& x, const Eigen::VectorXf& y, const IBoosterParams& params, uint count, const std::vector<string>& columnNames )noexcept(false);
	JDE_XGB_VISIBILITY BoosterParams Tune( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, uint testCount, const fs::path& saveStem, uint foldCount, const fs::path& paramStart )noexcept(false);
	JDE_XGB_VISIBILITY BoosterParams TuneOne( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, const fs::path& saveStem, uint foldCount, string_view parameterName )noexcept(false);


//Testing:
	std::tuple<uint,vector<double>> CrossValidate( const IBoosterParams& parameters, Eigen::MatrixXf& x, Math::Vector<>& y, uint foldCount, uint trainingRounds, string_view logSuffix  )noexcept(false);
#pragma region ParamResults
	struct ParamResults
	{
		ParamResults( const BoosterParams& parameters, uint bestIteration, const map<uint,vector<double>>& groupValues ):
			Parameters{ parameters },
			BestIteration{ bestIteration },
			GroupValues{ groupValues }
		{}
		ParamResults( const BoosterParams& parameters,double average, double stdDeviation, double min, double max, uint bestIteration ):
			Parameters{ parameters },
			BestIteration{bestIteration},
			_average{average},
			_stdDeviation{stdDeviation},
			_min{min},
			_max{max}
		{}
		const BoosterParams& Parameters;
		const uint BestIteration;
		const map<uint,vector<double>> GroupValues;
		double Average()const noexcept{ CalcResults(); return _average; }
		double StdDeviation()const noexcept{ CalcResults(); return _stdDeviation; }
		double Min()const noexcept{ CalcResults(); return _min; }
		double Max()const noexcept{ CalcResults(); return _max; }
		bool operator<( const ParamResults& other )const noexcept{ return Average()<other.Average(); }
	private:
		void CalcResults()const noexcept;
		mutable double _average{ std::numeric_limits<double>::max() };
		mutable double _stdDeviation{0.0};
		mutable double _min{0.0};
		mutable double _max{0.0};
	};
	BoosterParams SaveTuning( const fs::path& saveStem, const map<BoosterParams,ParamResults>& previousExecutions )noexcept(false);
#pragma endregion	
*/
}