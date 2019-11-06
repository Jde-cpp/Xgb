#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
/*
#ifndef _WINDOWS
	#define THREAD_LOCAL thread_local
#endif
*/
#pragma warning( disable : 4245) 
#include <boost/crc.hpp> 
#pragma warning( default : 4245) 
#include <boost/system/error_code.hpp>
#ifndef __INTELLISENSE__
	#include <spdlog/spdlog.h>
	#include <spdlog/sinks/basic_file_sink.h>
	#include <spdlog/fmt/ostr.h>
#endif

#include <xgboost/c_api.h>
//#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#pragma warning (disable:4275)
#pragma warning (disable:4251)
#include "../../Framework/source/TypeDefs.h"
#include "../../Framework/source/log/Logging.h"
#include "../../Framework/source/JdeAssert.h"
#include "../../Framework/source/Exception.h"
#include "../../Eigen/source/EMatrix.h"
#include "../../Framework/source/math/MathUtilities.h"

//#include "../../framework/StringUtilities.h"
//#include "../../framework/io/Buffer.h"
//#include "../../framework/math/EMatrix.h"
//#include "../../framework/math/MathUtilities.h"
//#include "../../framework/threading/Pool.h"
//#include "../../framework/Exception.h"
//#include "../../framework/Defines.h"
//#include "../../framework/Assert.h"
//#include "../../framework/Exception.h"

namespace Jde
{
	inline void Call( std::function<int()> func, std::string_view functionName )
	{
		const int result = func();
		if( result )
		{
			string error = XGBGetLastError();
			DBG0( error );
			THROW( Exception("{} - error ({}) - {}", functionName, result, error) );
		}
	}
}
//#define CALL(xgbFunction,functionName) std::function<int()> func = [&](){ return xgbFunction; }; Call( func, functionName )
#define CALL(xgbFunction,functionName) Call( [&](){ return xgbFunction; }, functionName )

