#include "Exports.h"
#include "../../DecisionTree/source/IDataset.h"

namespace Jde::AI::Dts
{
	struct IBoosterParams;
namespace Xgb
{
	typedef Eigen::Matrix<float,-1,-1,Eigen::RowMajor> XgbMatrix;
	typedef DMatrixHandle HDataset;
	struct JDE_XGB_VISIBILITY Dataset : public IDataset
	{
		Dataset( const Math::RowVector<float,-1>& y )noexcept(false);
		Dataset( const float* pFeatures, uint rows, uint cols )noexcept(false);
		
		Dataset( const XgbMatrix& matrix, const std::vector<string>* pColumnNames=nullptr )noexcept(false);
		Dataset( const XgbMatrix& matrix, const std::vector<string>* pColumnNames, const Eigen::VectorXf& y )noexcept(false);

		~Dataset();
		HDataset Handle()const{return _handle;}
		const std::vector<string>& Features()const noexcept{ return _features; } 
	private:
		HDataset _handle;
		std::vector<string> _features;
		//uint _maxFeatureLength{1024};
	};
}}	