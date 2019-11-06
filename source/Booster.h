#pragma once
#include "Exports.h"
#include "../../DecisionTree/source/IBooster.h"

namespace Jde::AI::Dts
{
	struct IBoosterParams;
	struct IDataset;
namespace Xgb
{
	typedef Eigen::Matrix<float,-1,-1,Eigen::RowMajor> XgbMatrix;
	typedef void* BoosterHandle;
	typedef void *DMatrixHandle;  // NOLINT(*)
	struct Dataset;
	class JDE_XGB_VISIBILITY Booster : public IBooster
	{
	public:
		//Booster( const BoosterParams& params, const Dataset& ds )noexcept(false);
		Booster( const IBoosterParams& params, sp<const IDataset>& pDS, sp<const IDataset> pValidation=nullptr )noexcept(false);
		Booster( const std::vector<char>& model )noexcept(false);
		Booster( const Booster& ) = delete;
		~Booster();
		Booster& operator=(const Booster&) = delete;

		bool UpdateOneIteration( int index=-1 )noexcept(false)override;
		//void LoadModelFromString( string_view model )noexcept(false)override;
		void LoadModelFromBinary( const vector<char>& bytes )noexcept(false);

		sp<vector<double>> Predict( const XgbMatrix& matrix )noexcept(false);
		double Predict( const Math::RowVector<float,-1>& vector )noexcept(false);
		double Predict( const double* pFeatures )noexcept(false) override;
		uint GetEvaluationCounts()const noexcept(false);
		vector<double> GetEvaluation( bool validation, uint iteration )const noexcept(false) override;

		MapPtr<string,double> FeatureImportances( EFeatureImportance featureImportance )const noexcept(false) override;
		bool HasCoverImportance()const noexcept override{ return true; }
		string to_string( uint iterationNumber = 0 )const noexcept(false) override;
		VectorPtr<string> StringTrees( bool withStats )const noexcept(false);
		uint FeatureCount()const noexcept(false);
		static uint SaveIfElse( const fs::path& modelPath, string_view namespaceName, ostream& osCpp )noexcept(false);
		static uint ModelCount( const fs::path& modelPath )noexcept(false);
		const vector<string>& FeatureNames()const noexcept;
		const uint TrainRowCount;
		const uint ValidationRowCount;
		double BestScore{ std::numeric_limits<double>::max() };
		constexpr static char FileSuffix[] = ".model";
		sp<const Dataset> ValidationPtr()const noexcept;
		void SetBestIteration( uint value )noexcept override;
		void LoadBestIteration()noexcept(false)override;
		const vector<char>& BestIterationModel()noexcept{ return _bestIterationValue; }
		void Save( const fs::path& path )noexcept(false) override;
	private:
		vector<float> Predict( const Dataset& ds );
		void Release()noexcept;
		sp<vector<double>> FeatureImportanceValues( bool splitImportance=true, int iterationNumber=-1 )const;
		vector<char> _bestIterationValue;
		mutable vector<string> _featureNames;
		BoosterHandle _handle{ nullptr };
		string GetAttribute( const string& name )const noexcept(false);
		void SetAttribute( const string& name, const string& value )noexcept(false);
		sp<const Dataset> _pValidation;
		sp<const Dataset> _pTrain;
	};
	typedef sp<Booster> BoosterPtr;
}}