#ifdef JDE_XGB_EXPORTS
	#ifdef _WINDOWS 
		#define JDE_XGB_VISIBILITY __declspec( dllexport )
	#else
		#define JDE_XGB_VISIBILITY __attribute__((visibility("default")))
	#endif
#else 
	#ifdef _WINDOWS 
		#define JDE_XGB_VISIBILITY __declspec( dllimport )
		//#define _GLIBCXX_USE_NOEXCEPT noexcept
		#if NDEBUG
			#pragma comment(lib, "Jde.AI.Xgb.lib")
		#else
			#pragma comment(lib, "Jde.AI.Xgb.lib")
		#endif
	#else
		#define JDE_XGB_VISIBILITY
	#endif
#endif 
