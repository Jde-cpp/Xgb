#include "stdafx.h"
#include "XgbTree.h"
//#include "../../framework/io/File.h"
#include "../../../framework/Defines.h"

namespace Jde::AI::Dts::Xgb
{
	map<uint,map<string,sp<Tree>>> _longTrees;
	map<uint,map<string,sp<Tree>>> _shortTrees;
//#ifdef TREE_NONHARDCODED
	sp<ITree> GetPrediction( const fs::path& path, uint16 minuteStart, bool isLong )
	{
		auto& trees = isLong ? _longTrees : _shortTrees;
		auto pMinuteTree = trees.try_emplace( minuteStart, map<string,sp<Tree>>{} ).first;
		auto pSymbolTrees = pMinuteTree->second.try_emplace( path.string(), shared_ptr<Tree>{} ).first;
		sp<ITree> pTree = pSymbolTrees->second;
		if( pTree==nullptr && fs::exists(path) )
		{
			try
			{
				pTree = pSymbolTrees->second = make_shared<Tree>( path );
			}
			catch( const Exception& exp )//no trees
			{
				WARN0( exp.what() );
			}
		}
		return pTree;
	}
	void ClearTrees( uint16_t minuteStart )
	{
		auto func = [minuteStart]( map<uint,map<string,sp<Tree>>>& trees )
		{
			auto pPrev = trees.end();
			for( auto pMinuteSymbols = trees.begin(); pMinuteSymbols!=trees.end(); ++pMinuteSymbols )
			{
				if( pMinuteSymbols->first<minuteStart && pPrev!=trees.end() )
				{
					trees.erase( pPrev );
					break;
				}
				pPrev = pMinuteSymbols;
			}
		};
		func( _longTrees );
		func( _shortTrees );
	}
//#endif TREE_NONHARDCODED

	Tree::Tree( const fs::path& path )noexcept(false):
		_booster( IO::FileUtilities::ToString(path) )
	{
		var& features = _booster.FeatureNames();
		_features.reserve( features.size() );
		for( var& feature: features )
			_features.push_back( feature );
	}

}