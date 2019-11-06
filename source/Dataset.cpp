#include "stdafx.h"
#include "Dataset.h"
#include <list>
#include "../../DecisionTree/source/IBoosterParams.h"
#define var const auto

namespace Jde::AI::Dts::Xgb
{
	Dataset::Dataset( const float* pFeatures, uint rows, uint cols )noexcept(false):
		IDataset{ rows, cols }
	{
		CALL( XGDMatrixCreateFromMat(pFeatures, static_cast<int>(RowCount), static_cast<int>(ColumnCount), NAN, &_handle), "XGDMatrixCreateFromMat" );
	}

	Dataset::Dataset( const Math::RowVector<float,-1>& matrix )noexcept(false):
		Dataset{ matrix.data(), 1, (uint)matrix.cols() }
	{}

	Dataset::Dataset( const XgbMatrix& matrix, const std::vector<string>* pColumnNames )noexcept(false):
		Dataset{ matrix.data(), (uint)matrix.rows(), (uint)matrix.cols() }
	{
		if( pColumnNames )
		{
			ASSRT_EQ( (uint)matrix.cols(), pColumnNames->size() );
			_features.reserve( pColumnNames->size() );
			std::for_each( pColumnNames->begin(), pColumnNames->end(), [&](auto& feature){_features.push_back(StringUtilities::Replace(feature, " ", "_" )); } );
		}
	}
	Dataset::Dataset( const XgbMatrix& matrix, const std::vector<string>* pColumnNames, const Eigen::VectorXf& y )noexcept(false):
		Dataset( matrix, pColumnNames )
	{
		ASSRT_EQ( matrix.rows(), y.rows() );
		CALL( XGDMatrixSetFloatInfo(_handle, "label", y.data(), y.rows()), "XGDMatrixSetFloatInfo" );
	} 

	Dataset::~Dataset()
	{
		var failed = XGDMatrixFree( _handle );
		if( failed )
			ERR( "({}) - {}", failed, XGBGetLastError() );
	}
}