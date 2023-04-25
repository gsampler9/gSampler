#ifndef GS_LOGGING_H_
#define GS_LOGGING_H_

#include <c10/util/Logging.h>

// Undefine macros to avoid conflicts between torch logger and glog
#undef LOG
#undef VLOG_IS_ON
#undef VLOG
#undef VLOG_IF
#undef LOG_IF
#undef CHECK
#undef CHECK_OP
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#undef CHECK_NOTNULL
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT
#undef DCHECK_NOTNULL
#include <glog/logging.h>

#endif