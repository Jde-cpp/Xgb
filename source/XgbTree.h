#pragma once
#include "Booster.h"
#include "Exports.h"
#include "../Tree.h"

namespace Jde::AI::Dts::Xgb
{
/*	struct ITree
	{
		template<class TTree>
		static ITree& GetInstance()noexcept;
		virtual double Predict( const double* pFeatures )noexcept=0;
		virtual const string_view* FeatureNames()const noexcept=0;
		virtual size_t FeatureCount()const noexcept=0;
		virtual MapPtr<string,double> FeatureImportances( bool splitImportance )const=0;
	};

	struct Tree : public ITree
	{
		Tree( const fs::path& path )noexcept(false);
		double Predict( const double* pFeatures )noexcept override{ return _booster.Predict(pFeatures); }
		const string_view* FeatureNames()const noexcept override{ return _features.data(); }
		size_t FeatureCount()const noexcept override{ return _features.size(); }
		map<string,int> FeatureImportance();
		MapPtr<string,double> FeatureImportances( bool splitImportance )const noexcept(false) override{ return _booster.FeatureImportances(splitImportance); }
	private:
		Booster _booster;
		vector<string_view> _features;
	};
*/
/*
	template<size_t TFeatureCount>
	struct TreeBase : public ITree
	{
		constexpr TreeBase( const std::array<string_view,TFeatureCount>& featureCount );
		const std::array<string_view,TFeatureCount> Features;

		const string_view* FeatureNames()const noexcept override{ return Features.data(); }
		size_t FeatureCount()const noexcept{return FeatureCnt;}
		constexpr static size_t FeatureCnt{TFeatureCount};
	};
	template<size_t TFeatureCount> 
	constexpr TreeBase<TFeatureCount>::TreeBase( const std::array<string_view,TFeatureCount>& features ):
		Features{features}
	{}
	*/
/*	template<class TTree>
	ITree& ITree::GetInstance()noexcept
	{
		if( TTree::_pInstance==nullptr )
			TTree::_pInstance = std::unique_ptr<TTree>( new TTree() );
		return *TTree::_pInstance;
	}
*/
	JDE_XGB_VISIBILITY sp<ITree> GetPrediction( const fs::path& path, uint16 minuteStart, bool isLong );
	void ClearTrees( uint16_t minuteStart );
}
