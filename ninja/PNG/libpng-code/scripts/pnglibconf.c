 PNG_DFN "/* 1.7.0 STANDARD API DEFINITION */" 
/*- pnglibconf.dfn intermediate file
 * generated from scripts/pnglibconf.dfa
 */
 PNG_DFN "/* pnglibconf.h - library build configuration */" 
 PNG_DFN "" 
 PNG_DFN "/* libpng version 1.7.0beta90, August 28, 2017 */" 
 PNG_DFN "" 
 PNG_DFN "/* Copyright (c) 1998-2016 Glenn Randers-Pehrson */" 
 PNG_DFN "" 
 PNG_DFN "/* This code is released under the libpng license. */" 
 PNG_DFN "/* For conditions of distribution and use, see the disclaimer */" 
 PNG_DFN "/* and license in png.h */" 
 PNG_DFN "" 
 PNG_DFN "/* pnglibconf.h */" 
 PNG_DFN "/* Machine generated file: DO NOT EDIT */" 
 PNG_DFN "/* Derived from: scripts/pnglibconf.dfa */" 
 PNG_DFN "#ifndef PNGLCONF_H" 
 PNG_DFN "#define PNGLCONF_H" 
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#if defined(HAVE_CONFIG_H) && !defined(PNG_NO_CONFIG_H)
# include "config.h"
#endif
#ifdef PNG_USER_CONFIG
# include "pngusr.h"
#endif
#ifdef __WATCOMC__
# ifndef PNG_API_RULE
# define PNG_API_RULE 2 /* Use Watcom calling conventions */
# endif
#endif
# include <zlib.h>
#ifdef PNG_SETJMP_NOT_SUPPORTED
#   define PNG_NO_SETJMP
#endif
#ifdef PNG_READ_TRANSFORMS_NOT_SUPPORTED
#   define PNG_NO_READ_TRANSFORMS
#endif
#ifdef PNG_NO_READ_COMPOSITED_NODIV
#   define PNG_NO_READ_COMPOSITE_NODIV
#endif
#ifdef PNG_INCH_CONVERSIONS
#   define PNG_INCH_CONVERSIONS_SUPPORTED
#endif
#ifdef PNG_WRITE_TRANSFORMS_NOT_SUPPORTED
#   define PNG_NO_WRITE_TRANSFORMS
#endif
#ifdef PNG_READ_ANCILLARY_CHUNKS_NOT_SUPPORTED
#   define PNG_NO_READ_ANCILLARY_CHUNKS
#endif
#ifdef PNG_WRITE_ANCILLARY_CHUNKS_NOT_SUPPORTED
#   define PNG_NO_WRITE_ANCILLARY_CHUNKS
#endif
#ifdef _WIN32_WCE
# define PNG_NO_CONVERT_tIME
#endif

/* OPTIONS */
 PNG_DFN "/* options */" 
PNG_DFN_START_SORT 2

/* option: CHECK_FOR_INVALID_INDEX enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_CHECK_FOR_INVALID_INDEX
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_CHECK_FOR_INVALID_INDEX_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED*/" 
#endif

/* option: ERROR_TEXT enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_ERROR_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_ERROR_TEXT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_ERROR_TEXT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_ERROR_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_ERROR_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_ERROR_TEXT_SUPPORTED*/" 
#endif

/* option: ALIGNED_MEMORY enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_ALIGNED_MEMORY_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_ALIGNED_MEMORY
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_ALIGNED_MEMORY_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_ALIGNED_MEMORY_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_ALIGNED_MEMORY_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_ALIGNED_MEMORY_SUPPORTED*/" 
#endif

/* option: READ enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_SUPPORTED*/" 
#endif

/* option: SETJMP enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SETJMP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SETJMP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SETJMP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SETJMP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SETJMP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SETJMP_SUPPORTED*/" 
#endif

/* option: FLOATING_ARITHMETIC enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_FLOATING_ARITHMETIC_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_FLOATING_ARITHMETIC
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_FLOATING_ARITHMETIC_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_FLOATING_ARITHMETIC_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_FLOATING_ARITHMETIC_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_FLOATING_ARITHMETIC_SUPPORTED*/" 
#endif

/* option: USER_LIMITS enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_USER_LIMITS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_USER_LIMITS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_USER_LIMITS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_USER_LIMITS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_USER_LIMITS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_USER_LIMITS_SUPPORTED*/" 
#endif

/* option: WARNINGS enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WARNINGS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WARNINGS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WARNINGS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WARNINGS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WARNINGS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WARNINGS_SUPPORTED*/" 
#endif

/* option: GET_PALETTE_MAX enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_GET_PALETTE_MAX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_GET_PALETTE_MAX
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_GET_PALETTE_MAX_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_GET_PALETTE_MAX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_GET_PALETTE_MAX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_GET_PALETTE_MAX_SUPPORTED*/" 
#endif

/* option: READ_COMPOSITE_NODIV enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_COMPOSITE_NODIV_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_COMPOSITE_NODIV
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_COMPOSITE_NODIV_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_COMPOSITE_NODIV_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_COMPOSITE_NODIV_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_COMPOSITE_NODIV_SUPPORTED*/" 
#endif

/* option: READ_INT_FUNCTIONS enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_INT_FUNCTIONS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_INT_FUNCTIONS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_INT_FUNCTIONS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_INT_FUNCTIONS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_INT_FUNCTIONS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_INT_FUNCTIONS_SUPPORTED*/" 
#endif

/* option: FLOATING_POINT enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_FLOATING_POINT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_FLOATING_POINT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_FLOATING_POINT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_FLOATING_POINT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_FLOATING_POINT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_FLOATING_POINT_SUPPORTED*/" 
#endif

/* option: READ_GET_PALETTE_MAX disabled
 *   requires:   READ
 *   if:        
 *   enabled-by: GET_PALETTE_MAX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by GET_PALETTE_MAX */
#ifdef PNG_GET_PALETTE_MAX_SUPPORTED
#   undef PNG_not_enabled /*GET_PALETTE_MAX*/
#endif
#   ifndef PNG_READ_GET_PALETTE_MAX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_GET_PALETTE_MAX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_GET_PALETTE_MAX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_GET_PALETTE_MAX_SUPPORTED*/" 
#endif

/* option: FIXED_POINT enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_FIXED_POINT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_FIXED_POINT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_FIXED_POINT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_FIXED_POINT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_FIXED_POINT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_FIXED_POINT_SUPPORTED*/" 
#endif

/* option: EASY_ACCESS enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_EASY_ACCESS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_EASY_ACCESS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_EASY_ACCESS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_EASY_ACCESS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_EASY_ACCESS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_EASY_ACCESS_SUPPORTED*/" 
#endif

/* option: SET_USER_LIMITS enabled
 *   requires:   USER_LIMITS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_USER_LIMITS_SUPPORTED
#   undef PNG_on /*!USER_LIMITS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SET_USER_LIMITS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SET_USER_LIMITS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SET_USER_LIMITS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SET_USER_LIMITS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SET_USER_LIMITS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SET_USER_LIMITS_SUPPORTED*/" 
#endif

/* option: STDIO enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_STDIO_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_STDIO
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_STDIO_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_STDIO_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_STDIO_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_STDIO_SUPPORTED*/" 
#endif

/* option: BENIGN_ERRORS enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_BENIGN_ERRORS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_BENIGN_ERRORS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_BENIGN_ERRORS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_BENIGN_ERRORS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_BENIGN_ERRORS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_BENIGN_ERRORS_SUPPORTED*/" 
#endif

/* option: UNKNOWN_CHUNKS enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_UNKNOWN_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_UNKNOWN_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_16BIT enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_16BIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_16BIT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_16BIT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_16BIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_16BIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_16BIT_SUPPORTED*/" 
#endif

/* option: MNG_READ_FEATURES enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_MNG_READ_FEATURES_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_MNG_READ_FEATURES
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_MNG_READ_FEATURES_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_MNG_READ_FEATURES_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_MNG_READ_FEATURES_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_MNG_READ_FEATURES_SUPPORTED*/" 
#endif

/* option: INCH_CONVERSIONS enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_INCH_CONVERSIONS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_INCH_CONVERSIONS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_INCH_CONVERSIONS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_INCH_CONVERSIONS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_INCH_CONVERSIONS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_INCH_CONVERSIONS_SUPPORTED*/" 
#endif

/* option: READ_TRANSFORMS enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_TRANSFORMS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_TRANSFORMS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_TRANSFORMS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_TRANSFORMS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_TRANSFORMS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_TRANSFORMS_SUPPORTED*/" 
#endif

/* option: READ_INTERLACING enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_INTERLACING_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_INTERLACING
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_INTERLACING_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_INTERLACING_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_INTERLACING_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_INTERLACING_SUPPORTED*/" 
#endif

/* option: READ_PACKSWAP enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_PACKSWAP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_PACKSWAP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_PACKSWAP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_PACKSWAP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_PACKSWAP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_PACKSWAP_SUPPORTED*/" 
#endif

/* option: SET_UNKNOWN_CHUNKS enabled
 *   requires:   UNKNOWN_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!UNKNOWN_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SET_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SET_UNKNOWN_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SET_UNKNOWN_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SET_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SET_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SET_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_STRIP_16_TO_8 enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_STRIP_16_TO_8_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_STRIP_16_TO_8
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_STRIP_16_TO_8_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_STRIP_16_TO_8_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_STRIP_16_TO_8_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_STRIP_16_TO_8_SUPPORTED*/" 
#endif

/* option: READ_BGR enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_BGR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_BGR
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_BGR_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_BGR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_BGR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_BGR_SUPPORTED*/" 
#endif

/* option: BUILD_GRAYSCALE_PALETTE enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_BUILD_GRAYSCALE_PALETTE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_BUILD_GRAYSCALE_PALETTE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_BUILD_GRAYSCALE_PALETTE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_BUILD_GRAYSCALE_PALETTE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_BUILD_GRAYSCALE_PALETTE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_BUILD_GRAYSCALE_PALETTE_SUPPORTED*/" 
#endif

/* option: USER_MEM enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_USER_MEM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_USER_MEM
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_USER_MEM_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_USER_MEM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_USER_MEM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_USER_MEM_SUPPORTED*/" 
#endif

/* option: READ_CHECK_FOR_INVALID_INDEX disabled
 *   requires:   READ
 *   if:        
 *   enabled-by: CHECK_FOR_INVALID_INDEX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by CHECK_FOR_INVALID_INDEX */
#ifdef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED
#   undef PNG_not_enabled /*CHECK_FOR_INVALID_INDEX*/
#endif
#   ifndef PNG_READ_CHECK_FOR_INVALID_INDEX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_CHECK_FOR_INVALID_INDEX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_CHECK_FOR_INVALID_INDEX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_CHECK_FOR_INVALID_INDEX_SUPPORTED*/" 
#endif

/* option: SEQUENTIAL_READ enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SEQUENTIAL_READ_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SEQUENTIAL_READ
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SEQUENTIAL_READ_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SEQUENTIAL_READ_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SEQUENTIAL_READ_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SEQUENTIAL_READ_SUPPORTED*/" 
#endif

/* option: MNG_WRITE_FEATURES enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_MNG_WRITE_FEATURES_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_MNG_WRITE_FEATURES
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_MNG_WRITE_FEATURES_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_MNG_WRITE_FEATURES_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_MNG_WRITE_FEATURES_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_MNG_WRITE_FEATURES_SUPPORTED*/" 
#endif

/* option: PROGRESSIVE_READ enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_PROGRESSIVE_READ_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_PROGRESSIVE_READ
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_PROGRESSIVE_READ_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_PROGRESSIVE_READ_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_PROGRESSIVE_READ_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_PROGRESSIVE_READ_SUPPORTED*/" 
#endif

/* option: READ_SWAP_ALPHA enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_SWAP_ALPHA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_SWAP_ALPHA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_SWAP_ALPHA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_SWAP_ALPHA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_SWAP_ALPHA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_SWAP_ALPHA_SUPPORTED*/" 
#endif

/* option: READ_ANCILLARY_CHUNKS enabled
 *   requires:   READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_ANCILLARY_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_ANCILLARY_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_ANCILLARY_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED*/" 
#endif

/* option: WRITE enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_SUPPORTED*/" 
#endif

/* option: HANDLE_AS_UNKNOWN enabled
 *   requires:   SET_UNKNOWN_CHUNKS READ
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SET_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!SET_UNKNOWN_CHUNKS*/
#endif
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_HANDLE_AS_UNKNOWN_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_HANDLE_AS_UNKNOWN
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_HANDLE_AS_UNKNOWN_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_HANDLE_AS_UNKNOWN_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_HANDLE_AS_UNKNOWN_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_HANDLE_AS_UNKNOWN_SUPPORTED*/" 
#endif

/* option: READ_bKGD enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_bKGD_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_bKGD
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_bKGD_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_bKGD_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_bKGD_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_bKGD_SUPPORTED*/" 
#endif

/* option: IO_STATE enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_IO_STATE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_IO_STATE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_IO_STATE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_IO_STATE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_IO_STATE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_IO_STATE_SUPPORTED*/" 
#endif

/* option: TIME_RFC1123 enabled
 *   requires:  
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_TIME_RFC1123_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_TIME_RFC1123
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_TIME_RFC1123_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_TIME_RFC1123_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_TIME_RFC1123_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_TIME_RFC1123_SUPPORTED*/" 
#endif

/* option: READ_FILLER enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_FILLER_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_FILLER
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_FILLER_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_FILLER_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_FILLER_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_FILLER_SUPPORTED*/" 
#endif

/* option: READ_sCAL enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_sCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_sCAL
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_sCAL_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_sCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_sCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_sCAL_SUPPORTED*/" 
#endif

/* option: READ_USER_TRANSFORM enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_USER_TRANSFORM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_USER_TRANSFORM
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_USER_TRANSFORM_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_USER_TRANSFORM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_USER_TRANSFORM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_USER_TRANSFORM_SUPPORTED*/" 
#endif

/* option: WRITE_CHECK_FOR_INVALID_INDEX disabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by: CHECK_FOR_INVALID_INDEX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by CHECK_FOR_INVALID_INDEX */
#ifdef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED
#   undef PNG_not_enabled /*CHECK_FOR_INVALID_INDEX*/
#endif
#   ifndef PNG_WRITE_CHECK_FOR_INVALID_INDEX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_CHECK_FOR_INVALID_INDEX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_CHECK_FOR_INVALID_INDEX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_CHECK_FOR_INVALID_INDEX_SUPPORTED*/" 
#endif

/* option: READ_USER_CHUNKS enabled
 *   requires:   READ UNKNOWN_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!UNKNOWN_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_USER_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_USER_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_USER_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_USER_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_USER_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_USER_CHUNKS_SUPPORTED*/" 
#endif

/* option: WRITE_TRANSFORMS enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_TRANSFORMS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_TRANSFORMS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_TRANSFORMS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_TRANSFORMS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_TRANSFORMS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_TRANSFORMS_SUPPORTED*/" 
#endif

/* option: READ_hIST enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_hIST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_hIST
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_hIST_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_hIST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_hIST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_hIST_SUPPORTED*/" 
#endif

/* option: WRITE_CUSTOMIZE_ZTXT_COMPRESSION enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_CUSTOMIZE_ZTXT_COMPRESSION
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED*/" 
#endif

/* option: READ_INVERT enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_INVERT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_INVERT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_INVERT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_INVERT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_INVERT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_INVERT_SUPPORTED*/" 
#endif

/* option: READ_tIME enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_tIME_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_tIME
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_tIME_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_tIME_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_tIME_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_tIME_SUPPORTED*/" 
#endif

/* option: CONSOLE_IO enabled
 *   requires:   STDIO
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_STDIO_SUPPORTED
#   undef PNG_on /*!STDIO*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_CONSOLE_IO_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_CONSOLE_IO
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_CONSOLE_IO_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_CONSOLE_IO_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_CONSOLE_IO_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_CONSOLE_IO_SUPPORTED*/" 
#endif

/* option: WRITE_UNKNOWN_CHUNKS enabled
 *   requires:   WRITE UNKNOWN_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!UNKNOWN_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_UNKNOWN_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_UNKNOWN_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_oFFs enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_oFFs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_oFFs
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_oFFs_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_oFFs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_oFFs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_oFFs_SUPPORTED*/" 
#endif

/* option: READ_iCCP enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_iCCP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_iCCP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_iCCP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_iCCP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_iCCP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_iCCP_SUPPORTED*/" 
#endif

/* option: READ_SWAP enabled
 *   requires:   READ_TRANSFORMS READ_16BIT
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_16BIT_SUPPORTED
#   undef PNG_on /*!READ_16BIT*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_SWAP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_SWAP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_SWAP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_SWAP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_SWAP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_SWAP_SUPPORTED*/" 
#endif

/* option: READ_pHYs enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_pHYs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_pHYs
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_pHYs_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_pHYs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_pHYs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_pHYs_SUPPORTED*/" 
#endif

/* option: WRITE_FILTER enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_FILTER_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_FILTER
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_FILTER_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_FILTER_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_FILTER_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_FILTER_SUPPORTED*/" 
#endif

/* option: READ_INVERT_ALPHA enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_INVERT_ALPHA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_INVERT_ALPHA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_INVERT_ALPHA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_INVERT_ALPHA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_INVERT_ALPHA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_INVERT_ALPHA_SUPPORTED*/" 
#endif

/* option: READ_sRGB enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_sRGB_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_sRGB
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_sRGB_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_sRGB_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_sRGB_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_sRGB_SUPPORTED*/" 
#endif

/* option: WRITE_PACKSWAP enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_PACKSWAP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_PACKSWAP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_PACKSWAP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_PACKSWAP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_PACKSWAP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_PACKSWAP_SUPPORTED*/" 
#endif

/* option: READ_PACK enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_PACK_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_PACK
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_PACK_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_PACK_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_PACK_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_PACK_SUPPORTED*/" 
#endif

/* option: WRITE_INTERLACING enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_INTERLACING_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_INTERLACING
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_INTERLACING_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_INTERLACING_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_INTERLACING_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_INTERLACING_SUPPORTED*/" 
#endif

/* option: READ_sPLT enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_sPLT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_sPLT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_sPLT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_sPLT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_sPLT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_sPLT_SUPPORTED*/" 
#endif

/* option: USER_CHUNKS disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_USER_CHUNKS
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_USER_CHUNKS */
#ifdef PNG_READ_USER_CHUNKS_SUPPORTED
#   undef PNG_not_enabled /*READ_USER_CHUNKS*/
#endif
#   ifndef PNG_USER_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_USER_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_USER_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_USER_CHUNKS_SUPPORTED*/" 
#endif

/* option: BENIGN_WRITE_ERRORS disabled
 *   requires:   BENIGN_ERRORS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_BENIGN_ERRORS_SUPPORTED
#   undef PNG_on /*!BENIGN_ERRORS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_BENIGN_WRITE_ERRORS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_BENIGN_WRITE_ERRORS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_BENIGN_WRITE_ERRORS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_BENIGN_WRITE_ERRORS_SUPPORTED*/" 
#endif

/* option: READ_cHRM enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_cHRM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_cHRM
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_cHRM_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_cHRM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_cHRM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_cHRM_SUPPORTED*/" 
#endif

/* option: WRITE_BGR enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_BGR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_BGR
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_BGR_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_BGR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_BGR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_BGR_SUPPORTED*/" 
#endif

/* option: WRITE_FLUSH enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_FLUSH_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_FLUSH
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_FLUSH_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_FLUSH_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_FLUSH_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_FLUSH_SUPPORTED*/" 
#endif

/* option: READ_sBIT enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_sBIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_sBIT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_sBIT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_sBIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_sBIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_sBIT_SUPPORTED*/" 
#endif

/* option: WRITE_WEIGHTED_FILTER enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_WEIGHTED_FILTER_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_WEIGHTED_FILTER
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_WEIGHTED_FILTER_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_WEIGHTED_FILTER_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_WEIGHTED_FILTER_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_WEIGHTED_FILTER_SUPPORTED*/" 
#endif

/* option: READ_IMAGE enabled
 *   requires:   SEQUENTIAL_READ READ_INTERLACING
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SEQUENTIAL_READ_SUPPORTED
#   undef PNG_on /*!SEQUENTIAL_READ*/
#endif
#ifndef PNG_READ_INTERLACING_SUPPORTED
#   undef PNG_on /*!READ_INTERLACING*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_IMAGE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_IMAGE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_IMAGE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_IMAGE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_IMAGE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_IMAGE_SUPPORTED*/" 
#endif

/* option: WRITE_INVERT enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_INVERT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_INVERT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_INVERT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_INVERT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_INVERT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_INVERT_SUPPORTED*/" 
#endif

/* option: BENIGN_READ_ERRORS enabled
 *   requires:   BENIGN_ERRORS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_BENIGN_ERRORS_SUPPORTED
#   undef PNG_on /*!BENIGN_ERRORS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_BENIGN_READ_ERRORS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_BENIGN_READ_ERRORS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_BENIGN_READ_ERRORS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_BENIGN_READ_ERRORS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_BENIGN_READ_ERRORS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_BENIGN_READ_ERRORS_SUPPORTED*/" 
#endif

/* option: READ_pCAL enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_pCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_pCAL
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_pCAL_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_pCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_pCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_pCAL_SUPPORTED*/" 
#endif

/* option: READ_OPT_PLTE enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_OPT_PLTE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_OPT_PLTE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_OPT_PLTE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_OPT_PLTE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_OPT_PLTE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_OPT_PLTE_SUPPORTED*/" 
#endif

/* option: READ_tRNS enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_tRNS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_tRNS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_tRNS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_tRNS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_tRNS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_tRNS_SUPPORTED*/" 
#endif

/* option: MNG_FEATURES disabled
 *   requires:  
 *   if:        
 *   enabled-by: MNG_READ_FEATURES MNG_WRITE_FEATURES
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by MNG_READ_FEATURES MNG_WRITE_FEATURES */
#ifdef PNG_MNG_READ_FEATURES_SUPPORTED
#   undef PNG_not_enabled /*MNG_READ_FEATURES*/
#endif
#ifdef PNG_MNG_WRITE_FEATURES_SUPPORTED
#   undef PNG_not_enabled /*MNG_WRITE_FEATURES*/
#endif
#   ifndef PNG_MNG_FEATURES_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_MNG_FEATURES_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_MNG_FEATURES_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_MNG_FEATURES_SUPPORTED*/" 
#endif

/* option: WRITE_PNG enabled
 *   requires:   WRITE WRITE_TRANSFORMS WRITE_INTERLACING
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#ifndef PNG_WRITE_INTERLACING_SUPPORTED
#   undef PNG_on /*!WRITE_INTERLACING*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_PNG_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_PNG
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_PNG_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_PNG_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_PNG_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_PNG_SUPPORTED*/" 
#endif

/* option: READ_QUANTIZE enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_QUANTIZE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_QUANTIZE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_QUANTIZE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_QUANTIZE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_QUANTIZE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_QUANTIZE_SUPPORTED*/" 
#endif

/* option: WRITE_GET_PALETTE_MAX disabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by: GET_PALETTE_MAX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by GET_PALETTE_MAX */
#ifdef PNG_GET_PALETTE_MAX_SUPPORTED
#   undef PNG_not_enabled /*GET_PALETTE_MAX*/
#endif
#   ifndef PNG_WRITE_GET_PALETTE_MAX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_GET_PALETTE_MAX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_GET_PALETTE_MAX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_GET_PALETTE_MAX_SUPPORTED*/" 
#endif

/* option: READ_STRIP_ALPHA enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_STRIP_ALPHA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_STRIP_ALPHA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_STRIP_ALPHA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_STRIP_ALPHA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_STRIP_ALPHA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_STRIP_ALPHA_SUPPORTED*/" 
#endif

/* option: READ_tEXt enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_tEXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_tEXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_tEXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_tEXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_tEXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_tEXt_SUPPORTED*/" 
#endif

/* option: WRITE_USER_TRANSFORM enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_USER_TRANSFORM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_USER_TRANSFORM
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_USER_TRANSFORM_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_USER_TRANSFORM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_USER_TRANSFORM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_USER_TRANSFORM_SUPPORTED*/" 
#endif

/* option: READ_SHIFT enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_SHIFT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_SHIFT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_SHIFT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_SHIFT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_SHIFT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_SHIFT_SUPPORTED*/" 
#endif

/* option: READ_iTXt enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_iTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_iTXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_iTXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_iTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_iTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_iTXt_SUPPORTED*/" 
#endif

/* option: WRITE_OPTIMIZE_CMF enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_OPTIMIZE_CMF_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_OPTIMIZE_CMF
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_OPTIMIZE_CMF_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_OPTIMIZE_CMF_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_OPTIMIZE_CMF_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_OPTIMIZE_CMF_SUPPORTED*/" 
#endif

/* option: SELECT_FILTER enabled
 *   requires:   WRITE_FILTER
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_FILTER_SUPPORTED
#   undef PNG_on /*!WRITE_FILTER*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SELECT_FILTER_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SELECT_FILTER
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SELECT_FILTER_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SELECT_FILTER_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SELECT_FILTER_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SELECT_FILTER_SUPPORTED*/" 
#endif

/* option: WRITE_SWAP_ALPHA enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_SWAP_ALPHA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_SWAP_ALPHA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_SWAP_ALPHA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_SWAP_ALPHA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_SWAP_ALPHA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_SWAP_ALPHA_SUPPORTED*/" 
#endif

/* option: WRITE_FILLER enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_FILLER_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_FILLER
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_FILLER_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_FILLER_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_FILLER_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_FILLER_SUPPORTED*/" 
#endif

/* option: WRITE_SHIFT enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_SHIFT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_SHIFT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_SHIFT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_SHIFT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_SHIFT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_SHIFT_SUPPORTED*/" 
#endif

/* option: WRITE_INT_FUNCTIONS disabled
 *   requires:  
 *   if:        
 *   enabled-by: WRITE
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by WRITE */
#ifdef PNG_WRITE_SUPPORTED
#   undef PNG_not_enabled /*WRITE*/
#endif
#   ifndef PNG_WRITE_INT_FUNCTIONS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_INT_FUNCTIONS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_INT_FUNCTIONS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_INT_FUNCTIONS_SUPPORTED*/" 
#endif

/* option: WRITE_16BIT enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_16BIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_16BIT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_16BIT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_16BIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_16BIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_16BIT_SUPPORTED*/" 
#endif

/* option: WRITE_INVERT_ALPHA enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_INVERT_ALPHA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_INVERT_ALPHA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_INVERT_ALPHA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_INVERT_ALPHA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_INVERT_ALPHA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_INVERT_ALPHA_SUPPORTED*/" 
#endif

/* option: WRITE_ANCILLARY_CHUNKS enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_ANCILLARY_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_gAMA enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_gAMA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_gAMA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_gAMA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_gAMA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_gAMA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_gAMA_SUPPORTED*/" 
#endif

/* option: WRITE_CUSTOMIZE_COMPRESSION enabled
 *   requires:   WRITE
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_CUSTOMIZE_COMPRESSION
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_CUSTOMIZE_COMPRESSION_SUPPORTED*/" 
#endif

/* option: WRITE_sCAL enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_sCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_sCAL
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_sCAL_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_sCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_sCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_sCAL_SUPPORTED*/" 
#endif

/* option: READ_zTXt enabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_zTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_zTXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_zTXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_zTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_zTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_zTXt_SUPPORTED*/" 
#endif

/* option: SAVE_UNKNOWN_CHUNKS enabled
 *   requires:   READ SET_UNKNOWN_CHUNKS HANDLE_AS_UNKNOWN
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#ifndef PNG_SET_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!SET_UNKNOWN_CHUNKS*/
#endif
#ifndef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
#   undef PNG_on /*!HANDLE_AS_UNKNOWN*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SAVE_UNKNOWN_CHUNKS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SAVE_UNKNOWN_CHUNKS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_COMPRESSED_TEXT disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_iCCP READ_iTXt READ_zTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_iCCP READ_iTXt READ_zTXt */
#ifdef PNG_READ_iCCP_SUPPORTED
#   undef PNG_not_enabled /*READ_iCCP*/
#endif
#ifdef PNG_READ_iTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_iTXt*/
#endif
#ifdef PNG_READ_zTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_zTXt*/
#endif
#   ifndef PNG_READ_COMPRESSED_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_COMPRESSED_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_COMPRESSED_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_COMPRESSED_TEXT_SUPPORTED*/" 
#endif

/* option: WRITE_gAMA enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_gAMA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_gAMA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_gAMA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_gAMA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_gAMA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_gAMA_SUPPORTED*/" 
#endif

/* option: WRITE_sRGB enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_sRGB_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_sRGB
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_sRGB_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_sRGB_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_sRGB_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_sRGB_SUPPORTED*/" 
#endif

/* option: CONVERT_tIME enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_CONVERT_tIME_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_CONVERT_tIME
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_CONVERT_tIME_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_CONVERT_tIME_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_CONVERT_tIME_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_CONVERT_tIME_SUPPORTED*/" 
#endif

/* option: WRITE_PACK enabled
 *   requires:   WRITE_TRANSFORMS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_PACK_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_PACK
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_PACK_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_PACK_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_PACK_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_PACK_SUPPORTED*/" 
#endif

/* option: WRITE_sBIT enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_sBIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_sBIT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_sBIT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_sBIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_sBIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_sBIT_SUPPORTED*/" 
#endif

/* option: 16BIT disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_16BIT WRITE_16BIT
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_16BIT WRITE_16BIT */
#ifdef PNG_READ_16BIT_SUPPORTED
#   undef PNG_not_enabled /*READ_16BIT*/
#endif
#ifdef PNG_WRITE_16BIT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_16BIT*/
#endif
#   ifndef PNG_16BIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_16BIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_16BIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_16BIT_SUPPORTED*/" 
#endif

/* option: WRITE_iCCP enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_iCCP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_iCCP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_iCCP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_iCCP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_iCCP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_iCCP_SUPPORTED*/" 
#endif

/* option: USER_TRANSFORM_INFO enabled
 *   requires:  
 *   if:         READ_USER_TRANSFORM WRITE_USER_TRANSFORM
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
/* if READ_USER_TRANSFORM WRITE_USER_TRANSFORM */
#define PNG_no_if 1
#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
#   undef PNG_no_if /*READ_USER_TRANSFORM*/
#endif
#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
#   undef PNG_no_if /*WRITE_USER_TRANSFORM*/
#endif
#ifdef PNG_no_if /*missing if*/
#   undef PNG_on
#endif
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_USER_TRANSFORM_INFO_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_USER_TRANSFORM_INFO
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_USER_TRANSFORM_INFO_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_USER_TRANSFORM_INFO_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_USER_TRANSFORM_INFO_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_USER_TRANSFORM_INFO_SUPPORTED*/" 
#endif

/* option: WRITE_bKGD enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_bKGD_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_bKGD
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_bKGD_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_bKGD_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_bKGD_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_bKGD_SUPPORTED*/" 
#endif

/* option: PALETTE_MAX disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_CHECK_FOR_INVALID_INDEX WRITE_CHECK_FOR_INVALID_INDEX READ_GET_PALETTE_MAX WRITE_GET_PALETTE_MAX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_CHECK_FOR_INVALID_INDEX WRITE_CHECK_FOR_INVALID_INDEX READ_GET_PALETTE_MAX WRITE_GET_PALETTE_MAX */
#ifdef PNG_READ_CHECK_FOR_INVALID_INDEX_SUPPORTED
#   undef PNG_not_enabled /*READ_CHECK_FOR_INVALID_INDEX*/
#endif
#ifdef PNG_WRITE_CHECK_FOR_INVALID_INDEX_SUPPORTED
#   undef PNG_not_enabled /*WRITE_CHECK_FOR_INVALID_INDEX*/
#endif
#ifdef PNG_READ_GET_PALETTE_MAX_SUPPORTED
#   undef PNG_not_enabled /*READ_GET_PALETTE_MAX*/
#endif
#ifdef PNG_WRITE_GET_PALETTE_MAX_SUPPORTED
#   undef PNG_not_enabled /*WRITE_GET_PALETTE_MAX*/
#endif
#   ifndef PNG_PALETTE_MAX_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_PALETTE_MAX_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_PALETTE_MAX_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_PALETTE_MAX_SUPPORTED*/" 
#endif

/* option: WRITE_sPLT enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_sPLT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_sPLT
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_sPLT_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_sPLT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_sPLT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_sPLT_SUPPORTED*/" 
#endif

/* option: WRITE_cHRM enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_cHRM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_cHRM
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_cHRM_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_cHRM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_cHRM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_cHRM_SUPPORTED*/" 
#endif

/* option: WRITE_iTXt enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_iTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_iTXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_iTXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_iTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_iTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_iTXt_SUPPORTED*/" 
#endif

/* option: WRITE_hIST enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_hIST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_hIST
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_hIST_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_hIST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_hIST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_hIST_SUPPORTED*/" 
#endif

/* option: WRITE_tEXt enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_tEXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_tEXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_tEXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_tEXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_tEXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_tEXt_SUPPORTED*/" 
#endif

/* option: gAMA disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_gAMA WRITE_gAMA
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_gAMA WRITE_gAMA */
#ifdef PNG_READ_gAMA_SUPPORTED
#   undef PNG_not_enabled /*READ_gAMA*/
#endif
#ifdef PNG_WRITE_gAMA_SUPPORTED
#   undef PNG_not_enabled /*WRITE_gAMA*/
#endif
#   ifndef PNG_gAMA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_gAMA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_gAMA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_gAMA_SUPPORTED*/" 
#endif

/* option: SET_OPTION disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ READ_sRGB WRITE_sRGB
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ READ_sRGB WRITE_sRGB */
#ifdef PNG_READ_SUPPORTED
#   undef PNG_not_enabled /*READ*/
#endif
#ifdef PNG_READ_sRGB_SUPPORTED
#   undef PNG_not_enabled /*READ_sRGB*/
#endif
#ifdef PNG_WRITE_sRGB_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sRGB*/
#endif
#   ifndef PNG_SET_OPTION_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SET_OPTION_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SET_OPTION_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SET_OPTION_SUPPORTED*/" 
#endif

/* option: STORE_UNKNOWN_CHUNKS disabled
 *   requires:   UNKNOWN_CHUNKS
 *   if:        
 *   enabled-by: WRITE_UNKNOWN_CHUNKS SAVE_UNKNOWN_CHUNKS
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!UNKNOWN_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by WRITE_UNKNOWN_CHUNKS SAVE_UNKNOWN_CHUNKS */
#ifdef PNG_WRITE_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_not_enabled /*WRITE_UNKNOWN_CHUNKS*/
#endif
#ifdef PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_not_enabled /*SAVE_UNKNOWN_CHUNKS*/
#endif
#   ifndef PNG_STORE_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_STORE_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_STORE_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_STORE_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: READ_PNG enabled
 *   requires:   READ_IMAGE READ_TRANSFORMS READ_INTERLACING
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_IMAGE_SUPPORTED
#   undef PNG_on /*!READ_IMAGE*/
#endif
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_INTERLACING_SUPPORTED
#   undef PNG_on /*!READ_INTERLACING*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_PNG_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_PNG
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_PNG_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_PNG_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_PNG_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_PNG_SUPPORTED*/" 
#endif

/* option: INFO_IMAGE disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_PNG WRITE_PNG
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_PNG WRITE_PNG */
#ifdef PNG_READ_PNG_SUPPORTED
#   undef PNG_not_enabled /*READ_PNG*/
#endif
#ifdef PNG_WRITE_PNG_SUPPORTED
#   undef PNG_not_enabled /*WRITE_PNG*/
#endif
#   ifndef PNG_INFO_IMAGE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_INFO_IMAGE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_INFO_IMAGE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_INFO_IMAGE_SUPPORTED*/" 
#endif

/* option: iTXt disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_iTXt WRITE_iTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_iTXt WRITE_iTXt */
#ifdef PNG_READ_iTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_iTXt*/
#endif
#ifdef PNG_WRITE_iTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iTXt*/
#endif
#   ifndef PNG_iTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_iTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_iTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_iTXt_SUPPORTED*/" 
#endif

/* option: WRITE_pHYs enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_pHYs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_pHYs
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_pHYs_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_pHYs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_pHYs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_pHYs_SUPPORTED*/" 
#endif

/* option: bKGD disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_bKGD WRITE_bKGD
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_bKGD WRITE_bKGD */
#ifdef PNG_READ_bKGD_SUPPORTED
#   undef PNG_not_enabled /*READ_bKGD*/
#endif
#ifdef PNG_WRITE_bKGD_SUPPORTED
#   undef PNG_not_enabled /*WRITE_bKGD*/
#endif
#   ifndef PNG_bKGD_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_bKGD_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_bKGD_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_bKGD_SUPPORTED*/" 
#endif

/* option: WRITE_SWAP enabled
 *   requires:   WRITE_TRANSFORMS WRITE_16BIT
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!WRITE_TRANSFORMS*/
#endif
#ifndef PNG_WRITE_16BIT_SUPPORTED
#   undef PNG_on /*!WRITE_16BIT*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_SWAP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_SWAP
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_SWAP_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_SWAP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_SWAP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_SWAP_SUPPORTED*/" 
#endif

/* option: WRITE_zTXt enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_zTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_zTXt
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_zTXt_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_zTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_zTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_zTXt_SUPPORTED*/" 
#endif

/* option: USER_TRANSFORM_PTR enabled
 *   requires:  
 *   if:         READ_USER_TRANSFORM WRITE_USER_TRANSFORM
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
/* if READ_USER_TRANSFORM WRITE_USER_TRANSFORM */
#define PNG_no_if 1
#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
#   undef PNG_no_if /*READ_USER_TRANSFORM*/
#endif
#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
#   undef PNG_no_if /*WRITE_USER_TRANSFORM*/
#endif
#ifdef PNG_no_if /*missing if*/
#   undef PNG_on
#endif
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_USER_TRANSFORM_PTR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_USER_TRANSFORM_PTR
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_USER_TRANSFORM_PTR_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_USER_TRANSFORM_PTR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_USER_TRANSFORM_PTR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_USER_TRANSFORM_PTR_SUPPORTED*/" 
#endif

/* option: READ_TEXT disabled
 *   requires:   READ_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by: READ_tEXt READ_zTXt READ_iTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!READ_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_tEXt READ_zTXt READ_iTXt */
#ifdef PNG_READ_tEXt_SUPPORTED
#   undef PNG_not_enabled /*READ_tEXt*/
#endif
#ifdef PNG_READ_zTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_zTXt*/
#endif
#ifdef PNG_READ_iTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_iTXt*/
#endif
#   ifndef PNG_READ_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_TEXT_SUPPORTED*/" 
#endif

/* option: sBIT disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_sBIT WRITE_sBIT
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_sBIT WRITE_sBIT */
#ifdef PNG_READ_sBIT_SUPPORTED
#   undef PNG_not_enabled /*READ_sBIT*/
#endif
#ifdef PNG_WRITE_sBIT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sBIT*/
#endif
#   ifndef PNG_sBIT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_sBIT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_sBIT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_sBIT_SUPPORTED*/" 
#endif

/* option: sCAL disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_sCAL WRITE_sCAL
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_sCAL WRITE_sCAL */
#ifdef PNG_READ_sCAL_SUPPORTED
#   undef PNG_not_enabled /*READ_sCAL*/
#endif
#ifdef PNG_WRITE_sCAL_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sCAL*/
#endif
#   ifndef PNG_sCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_sCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_sCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_sCAL_SUPPORTED*/" 
#endif

/* option: READ_UNKNOWN_CHUNKS disabled
 *   requires:   READ UNKNOWN_CHUNKS
 *   if:        
 *   enabled-by: SAVE_UNKNOWN_CHUNKS READ_USER_CHUNKS
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_SUPPORTED
#   undef PNG_on /*!READ*/
#endif
#ifndef PNG_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_on /*!UNKNOWN_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by SAVE_UNKNOWN_CHUNKS READ_USER_CHUNKS */
#ifdef PNG_SAVE_UNKNOWN_CHUNKS_SUPPORTED
#   undef PNG_not_enabled /*SAVE_UNKNOWN_CHUNKS*/
#endif
#ifdef PNG_READ_USER_CHUNKS_SUPPORTED
#   undef PNG_not_enabled /*READ_USER_CHUNKS*/
#endif
#   ifndef PNG_READ_UNKNOWN_CHUNKS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_UNKNOWN_CHUNKS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_UNKNOWN_CHUNKS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_UNKNOWN_CHUNKS_SUPPORTED*/" 
#endif

/* option: WRITE_tRNS enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_tRNS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_tRNS
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_tRNS_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_tRNS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_tRNS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_tRNS_SUPPORTED*/" 
#endif

/* option: WRITE_tIME enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_tIME_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_tIME
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_tIME_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_tIME_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_tIME_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_tIME_SUPPORTED*/" 
#endif

/* option: WRITE_pCAL enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_pCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_pCAL
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_pCAL_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_pCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_pCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_pCAL_SUPPORTED*/" 
#endif

/* option: sRGB disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_sRGB WRITE_sRGB
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_sRGB WRITE_sRGB */
#ifdef PNG_READ_sRGB_SUPPORTED
#   undef PNG_not_enabled /*READ_sRGB*/
#endif
#ifdef PNG_WRITE_sRGB_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sRGB*/
#endif
#   ifndef PNG_sRGB_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_sRGB_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_sRGB_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_sRGB_SUPPORTED*/" 
#endif

/* option: WRITE_oFFs enabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_WRITE_oFFs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_WRITE_oFFs
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_WRITE_oFFs_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_oFFs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_oFFs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_oFFs_SUPPORTED*/" 
#endif

/* option: READ_GAMMA enabled
 *   requires:   READ_TRANSFORMS READ_gAMA READ_sRGB
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_gAMA_SUPPORTED
#   undef PNG_on /*!READ_gAMA*/
#endif
#ifndef PNG_READ_sRGB_SUPPORTED
#   undef PNG_on /*!READ_sRGB*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_GAMMA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_GAMMA
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_GAMMA_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_GAMMA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_GAMMA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_GAMMA_SUPPORTED*/" 
#endif

/* option: cHRM disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_cHRM WRITE_cHRM
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_cHRM WRITE_cHRM */
#ifdef PNG_READ_cHRM_SUPPORTED
#   undef PNG_not_enabled /*READ_cHRM*/
#endif
#ifdef PNG_WRITE_cHRM_SUPPORTED
#   undef PNG_not_enabled /*WRITE_cHRM*/
#endif
#   ifndef PNG_cHRM_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_cHRM_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_cHRM_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_cHRM_SUPPORTED*/" 
#endif

/* option: oFFs disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_oFFs WRITE_oFFs
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_oFFs WRITE_oFFs */
#ifdef PNG_READ_oFFs_SUPPORTED
#   undef PNG_not_enabled /*READ_oFFs*/
#endif
#ifdef PNG_WRITE_oFFs_SUPPORTED
#   undef PNG_not_enabled /*WRITE_oFFs*/
#endif
#   ifndef PNG_oFFs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_oFFs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_oFFs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_oFFs_SUPPORTED*/" 
#endif

/* option: tEXt disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_tEXt WRITE_tEXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_tEXt WRITE_tEXt */
#ifdef PNG_READ_tEXt_SUPPORTED
#   undef PNG_not_enabled /*READ_tEXt*/
#endif
#ifdef PNG_WRITE_tEXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_tEXt*/
#endif
#   ifndef PNG_tEXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_tEXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_tEXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_tEXt_SUPPORTED*/" 
#endif

/* option: tIME disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_tIME WRITE_tIME
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_tIME WRITE_tIME */
#ifdef PNG_READ_tIME_SUPPORTED
#   undef PNG_not_enabled /*READ_tIME*/
#endif
#ifdef PNG_WRITE_tIME_SUPPORTED
#   undef PNG_not_enabled /*WRITE_tIME*/
#endif
#   ifndef PNG_tIME_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_tIME_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_tIME_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_tIME_SUPPORTED*/" 
#endif

/* option: pHYs disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_pHYs WRITE_pHYs
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_pHYs WRITE_pHYs */
#ifdef PNG_READ_pHYs_SUPPORTED
#   undef PNG_not_enabled /*READ_pHYs*/
#endif
#ifdef PNG_WRITE_pHYs_SUPPORTED
#   undef PNG_not_enabled /*WRITE_pHYs*/
#endif
#   ifndef PNG_pHYs_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_pHYs_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_pHYs_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_pHYs_SUPPORTED*/" 
#endif

/* option: READ_BACKGROUND enabled
 *   requires:   READ_TRANSFORMS READ_STRIP_ALPHA READ_GAMMA READ_tRNS
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_STRIP_ALPHA_SUPPORTED
#   undef PNG_on /*!READ_STRIP_ALPHA*/
#endif
#ifndef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_on /*!READ_GAMMA*/
#endif
#ifndef PNG_READ_tRNS_SUPPORTED
#   undef PNG_on /*!READ_tRNS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_BACKGROUND_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_BACKGROUND
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_BACKGROUND_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_BACKGROUND_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_BACKGROUND_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_BACKGROUND_SUPPORTED*/" 
#endif

/* option: iCCP disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_iCCP WRITE_iCCP
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_iCCP WRITE_iCCP */
#ifdef PNG_READ_iCCP_SUPPORTED
#   undef PNG_not_enabled /*READ_iCCP*/
#endif
#ifdef PNG_WRITE_iCCP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iCCP*/
#endif
#   ifndef PNG_iCCP_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_iCCP_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_iCCP_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_iCCP_SUPPORTED*/" 
#endif

/* option: READ_EXPAND enabled
 *   requires:   READ_TRANSFORMS READ_tRNS
 *   if:        
 *   enabled-by: READ_GAMMA
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_tRNS_SUPPORTED
#   undef PNG_on /*!READ_tRNS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_GAMMA */
#ifdef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_not_enabled /*READ_GAMMA*/
#endif
#   ifndef PNG_READ_EXPAND_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_EXPAND
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_EXPAND_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_EXPAND_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_EXPAND_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_EXPAND_SUPPORTED*/" 
#endif

/* option: tRNS disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_tRNS WRITE_tRNS
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_tRNS WRITE_tRNS */
#ifdef PNG_READ_tRNS_SUPPORTED
#   undef PNG_not_enabled /*READ_tRNS*/
#endif
#ifdef PNG_WRITE_tRNS_SUPPORTED
#   undef PNG_not_enabled /*WRITE_tRNS*/
#endif
#   ifndef PNG_tRNS_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_tRNS_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_tRNS_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_tRNS_SUPPORTED*/" 
#endif

/* option: hIST disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_hIST WRITE_hIST
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_hIST WRITE_hIST */
#ifdef PNG_READ_hIST_SUPPORTED
#   undef PNG_not_enabled /*READ_hIST*/
#endif
#ifdef PNG_WRITE_hIST_SUPPORTED
#   undef PNG_not_enabled /*WRITE_hIST*/
#endif
#   ifndef PNG_hIST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_hIST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_hIST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_hIST_SUPPORTED*/" 
#endif

/* option: sPLT disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_sPLT WRITE_sPLT
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_sPLT WRITE_sPLT */
#ifdef PNG_READ_sPLT_SUPPORTED
#   undef PNG_not_enabled /*READ_sPLT*/
#endif
#ifdef PNG_WRITE_sPLT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sPLT*/
#endif
#   ifndef PNG_sPLT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_sPLT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_sPLT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_sPLT_SUPPORTED*/" 
#endif

/* option: READ_ALPHA_MODE enabled
 *   requires:   READ_TRANSFORMS READ_GAMMA READ_BACKGROUND
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_on /*!READ_GAMMA*/
#endif
#ifndef PNG_READ_BACKGROUND_SUPPORTED
#   undef PNG_on /*!READ_BACKGROUND*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_ALPHA_MODE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_ALPHA_MODE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_ALPHA_MODE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_ALPHA_MODE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_ALPHA_MODE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_ALPHA_MODE_SUPPORTED*/" 
#endif

/* option: WRITE_COMPRESSED_TEXT disabled
 *   requires:  
 *   if:        
 *   enabled-by: WRITE_iCCP WRITE_iTXt WRITE_zTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by WRITE_iCCP WRITE_iTXt WRITE_zTXt */
#ifdef PNG_WRITE_iCCP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iCCP*/
#endif
#ifdef PNG_WRITE_iTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iTXt*/
#endif
#ifdef PNG_WRITE_zTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_zTXt*/
#endif
#   ifndef PNG_WRITE_COMPRESSED_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_COMPRESSED_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_COMPRESSED_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_COMPRESSED_TEXT_SUPPORTED*/" 
#endif

/* option: READ_GRAY_TO_RGB enabled
 *   requires:   READ_TRANSFORMS READ_EXPAND
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_EXPAND_SUPPORTED
#   undef PNG_on /*!READ_EXPAND*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_GRAY_TO_RGB_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_GRAY_TO_RGB
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_GRAY_TO_RGB_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_GRAY_TO_RGB_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_GRAY_TO_RGB_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_GRAY_TO_RGB_SUPPORTED*/" 
#endif

/* option: zTXt disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_zTXt WRITE_zTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_zTXt WRITE_zTXt */
#ifdef PNG_READ_zTXt_SUPPORTED
#   undef PNG_not_enabled /*READ_zTXt*/
#endif
#ifdef PNG_WRITE_zTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_zTXt*/
#endif
#   ifndef PNG_zTXt_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_zTXt_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_zTXt_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_zTXt_SUPPORTED*/" 
#endif

/* option: WRITE_TEXT disabled
 *   requires:   WRITE_ANCILLARY_CHUNKS
 *   if:        
 *   enabled-by: WRITE_tEXt WRITE_zTXt WRITE_iTXt
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_ANCILLARY_CHUNKS_SUPPORTED
#   undef PNG_on /*!WRITE_ANCILLARY_CHUNKS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by WRITE_tEXt WRITE_zTXt WRITE_iTXt */
#ifdef PNG_WRITE_tEXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_tEXt*/
#endif
#ifdef PNG_WRITE_zTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_zTXt*/
#endif
#ifdef PNG_WRITE_iTXt_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iTXt*/
#endif
#   ifndef PNG_WRITE_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_WRITE_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_WRITE_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_WRITE_TEXT_SUPPORTED*/" 
#endif

/* option: pCAL disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_pCAL WRITE_pCAL
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_pCAL WRITE_pCAL */
#ifdef PNG_READ_pCAL_SUPPORTED
#   undef PNG_not_enabled /*READ_pCAL*/
#endif
#ifdef PNG_WRITE_pCAL_SUPPORTED
#   undef PNG_not_enabled /*WRITE_pCAL*/
#endif
#   ifndef PNG_pCAL_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_pCAL_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_pCAL_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_pCAL_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_WRITE enabled
 *   requires:   WRITE SETJMP WRITE_SWAP WRITE_PACK WRITE_tRNS WRITE_gAMA WRITE_sRGB WRITE_cHRM
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_WRITE_SUPPORTED
#   undef PNG_on /*!WRITE*/
#endif
#ifndef PNG_SETJMP_SUPPORTED
#   undef PNG_on /*!SETJMP*/
#endif
#ifndef PNG_WRITE_SWAP_SUPPORTED
#   undef PNG_on /*!WRITE_SWAP*/
#endif
#ifndef PNG_WRITE_PACK_SUPPORTED
#   undef PNG_on /*!WRITE_PACK*/
#endif
#ifndef PNG_WRITE_tRNS_SUPPORTED
#   undef PNG_on /*!WRITE_tRNS*/
#endif
#ifndef PNG_WRITE_gAMA_SUPPORTED
#   undef PNG_on /*!WRITE_gAMA*/
#endif
#ifndef PNG_WRITE_sRGB_SUPPORTED
#   undef PNG_on /*!WRITE_sRGB*/
#endif
#ifndef PNG_WRITE_cHRM_SUPPORTED
#   undef PNG_on /*!WRITE_cHRM*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_WRITE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_WRITE
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_WRITE_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_WRITE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_WRITE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_WRITE_SUPPORTED*/" 
#endif

/* option: READ_RGB_TO_GRAY enabled
 *   requires:   READ_TRANSFORMS READ_GAMMA
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_on /*!READ_GAMMA*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_RGB_TO_GRAY_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_RGB_TO_GRAY
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_RGB_TO_GRAY_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_RGB_TO_GRAY_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_RGB_TO_GRAY_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_RGB_TO_GRAY_SUPPORTED*/" 
#endif

/* option: TEXT disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_TEXT WRITE_TEXT
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_TEXT WRITE_TEXT */
#ifdef PNG_READ_TEXT_SUPPORTED
#   undef PNG_not_enabled /*READ_TEXT*/
#endif
#ifdef PNG_WRITE_TEXT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_TEXT*/
#endif
#   ifndef PNG_TEXT_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_TEXT_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_TEXT_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_TEXT_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_WRITE_AFIRST enabled
 *   requires:   SIMPLIFIED_WRITE WRITE_SWAP_ALPHA
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SIMPLIFIED_WRITE_SUPPORTED
#   undef PNG_on /*!SIMPLIFIED_WRITE*/
#endif
#ifndef PNG_WRITE_SWAP_ALPHA_SUPPORTED
#   undef PNG_on /*!WRITE_SWAP_ALPHA*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_WRITE_AFIRST
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_WRITE_AFIRST_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED*/" 
#endif

/* option: READ_EXPAND_16 enabled
 *   requires:   READ_TRANSFORMS READ_16BIT READ_EXPAND
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_READ_16BIT_SUPPORTED
#   undef PNG_on /*!READ_16BIT*/
#endif
#ifndef PNG_READ_EXPAND_SUPPORTED
#   undef PNG_on /*!READ_EXPAND*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_READ_EXPAND_16_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_EXPAND_16
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_EXPAND_16_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_EXPAND_16_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_EXPAND_16_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_EXPAND_16_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_WRITE_BGR enabled
 *   requires:   SIMPLIFIED_WRITE WRITE_BGR
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SIMPLIFIED_WRITE_SUPPORTED
#   undef PNG_on /*!SIMPLIFIED_WRITE*/
#endif
#ifndef PNG_WRITE_BGR_SUPPORTED
#   undef PNG_on /*!WRITE_BGR*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_WRITE_BGR
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_WRITE_BGR_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED*/" 
#endif

/* option: COLORSPACE disabled
 *   requires:  
 *   if:        
 *   enabled-by: READ_RGB_TO_GRAY READ_cHRM WRITE_cHRM READ_iCCP WRITE_iCCP READ_sRGB WRITE_sRGB
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_RGB_TO_GRAY READ_cHRM WRITE_cHRM READ_iCCP WRITE_iCCP READ_sRGB WRITE_sRGB */
#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
#   undef PNG_not_enabled /*READ_RGB_TO_GRAY*/
#endif
#ifdef PNG_READ_cHRM_SUPPORTED
#   undef PNG_not_enabled /*READ_cHRM*/
#endif
#ifdef PNG_WRITE_cHRM_SUPPORTED
#   undef PNG_not_enabled /*WRITE_cHRM*/
#endif
#ifdef PNG_READ_iCCP_SUPPORTED
#   undef PNG_not_enabled /*READ_iCCP*/
#endif
#ifdef PNG_WRITE_iCCP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iCCP*/
#endif
#ifdef PNG_READ_sRGB_SUPPORTED
#   undef PNG_not_enabled /*READ_sRGB*/
#endif
#ifdef PNG_WRITE_sRGB_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sRGB*/
#endif
#   ifndef PNG_COLORSPACE_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_COLORSPACE_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_COLORSPACE_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_COLORSPACE_SUPPORTED*/" 
#endif

/* option: READ_SCALE_16_TO_8 enabled
 *   requires:   READ_TRANSFORMS
 *   if:        
 *   enabled-by: READ_GAMMA READ_RGB_TO_GRAY
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by READ_GAMMA READ_RGB_TO_GRAY */
#ifdef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_not_enabled /*READ_GAMMA*/
#endif
#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
#   undef PNG_not_enabled /*READ_RGB_TO_GRAY*/
#endif
#   ifndef PNG_READ_SCALE_16_TO_8_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_READ_SCALE_16_TO_8
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_READ_SCALE_16_TO_8_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_READ_SCALE_16_TO_8_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_READ_SCALE_16_TO_8_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_READ_SCALE_16_TO_8_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_WRITE_STDIO enabled
 *   requires:   SIMPLIFIED_WRITE STDIO
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SIMPLIFIED_WRITE_SUPPORTED
#   undef PNG_on /*!SIMPLIFIED_WRITE*/
#endif
#ifndef PNG_STDIO_SUPPORTED
#   undef PNG_on /*!STDIO*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_WRITE_STDIO_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_WRITE_STDIO
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_WRITE_STDIO_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_WRITE_STDIO_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_WRITE_STDIO_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_WRITE_STDIO_SUPPORTED*/" 
#endif

/* option: GAMMA disabled
 *   requires:  
 *   if:        
 *   enabled-by: COLORSPACE READ_gAMA WRITE_gAMA READ_iCCP WRITE_iCCP READ_sRGB WRITE_sRGB
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by COLORSPACE READ_gAMA WRITE_gAMA READ_iCCP WRITE_iCCP READ_sRGB WRITE_sRGB */
#ifdef PNG_COLORSPACE_SUPPORTED
#   undef PNG_not_enabled /*COLORSPACE*/
#endif
#ifdef PNG_READ_gAMA_SUPPORTED
#   undef PNG_not_enabled /*READ_gAMA*/
#endif
#ifdef PNG_WRITE_gAMA_SUPPORTED
#   undef PNG_not_enabled /*WRITE_gAMA*/
#endif
#ifdef PNG_READ_iCCP_SUPPORTED
#   undef PNG_not_enabled /*READ_iCCP*/
#endif
#ifdef PNG_WRITE_iCCP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_iCCP*/
#endif
#ifdef PNG_READ_sRGB_SUPPORTED
#   undef PNG_not_enabled /*READ_sRGB*/
#endif
#ifdef PNG_WRITE_sRGB_SUPPORTED
#   undef PNG_not_enabled /*WRITE_sRGB*/
#endif
#   ifndef PNG_GAMMA_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_GAMMA_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_GAMMA_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_GAMMA_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_READ enabled
 *   requires:   SEQUENTIAL_READ READ_TRANSFORMS SETJMP BENIGN_ERRORS READ_EXPAND READ_16BIT READ_EXPAND_16 READ_SCALE_16_TO_8 READ_RGB_TO_GRAY READ_ALPHA_MODE READ_BACKGROUND READ_STRIP_ALPHA READ_FILLER READ_SWAP READ_PACK READ_GRAY_TO_RGB READ_GAMMA READ_tRNS READ_bKGD READ_gAMA READ_cHRM READ_sRGB READ_sBIT READ_INTERLACING
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SEQUENTIAL_READ_SUPPORTED
#   undef PNG_on /*!SEQUENTIAL_READ*/
#endif
#ifndef PNG_READ_TRANSFORMS_SUPPORTED
#   undef PNG_on /*!READ_TRANSFORMS*/
#endif
#ifndef PNG_SETJMP_SUPPORTED
#   undef PNG_on /*!SETJMP*/
#endif
#ifndef PNG_BENIGN_ERRORS_SUPPORTED
#   undef PNG_on /*!BENIGN_ERRORS*/
#endif
#ifndef PNG_READ_EXPAND_SUPPORTED
#   undef PNG_on /*!READ_EXPAND*/
#endif
#ifndef PNG_READ_16BIT_SUPPORTED
#   undef PNG_on /*!READ_16BIT*/
#endif
#ifndef PNG_READ_EXPAND_16_SUPPORTED
#   undef PNG_on /*!READ_EXPAND_16*/
#endif
#ifndef PNG_READ_SCALE_16_TO_8_SUPPORTED
#   undef PNG_on /*!READ_SCALE_16_TO_8*/
#endif
#ifndef PNG_READ_RGB_TO_GRAY_SUPPORTED
#   undef PNG_on /*!READ_RGB_TO_GRAY*/
#endif
#ifndef PNG_READ_ALPHA_MODE_SUPPORTED
#   undef PNG_on /*!READ_ALPHA_MODE*/
#endif
#ifndef PNG_READ_BACKGROUND_SUPPORTED
#   undef PNG_on /*!READ_BACKGROUND*/
#endif
#ifndef PNG_READ_STRIP_ALPHA_SUPPORTED
#   undef PNG_on /*!READ_STRIP_ALPHA*/
#endif
#ifndef PNG_READ_FILLER_SUPPORTED
#   undef PNG_on /*!READ_FILLER*/
#endif
#ifndef PNG_READ_SWAP_SUPPORTED
#   undef PNG_on /*!READ_SWAP*/
#endif
#ifndef PNG_READ_PACK_SUPPORTED
#   undef PNG_on /*!READ_PACK*/
#endif
#ifndef PNG_READ_GRAY_TO_RGB_SUPPORTED
#   undef PNG_on /*!READ_GRAY_TO_RGB*/
#endif
#ifndef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_on /*!READ_GAMMA*/
#endif
#ifndef PNG_READ_tRNS_SUPPORTED
#   undef PNG_on /*!READ_tRNS*/
#endif
#ifndef PNG_READ_bKGD_SUPPORTED
#   undef PNG_on /*!READ_bKGD*/
#endif
#ifndef PNG_READ_gAMA_SUPPORTED
#   undef PNG_on /*!READ_gAMA*/
#endif
#ifndef PNG_READ_cHRM_SUPPORTED
#   undef PNG_on /*!READ_cHRM*/
#endif
#ifndef PNG_READ_sRGB_SUPPORTED
#   undef PNG_on /*!READ_sRGB*/
#endif
#ifndef PNG_READ_sBIT_SUPPORTED
#   undef PNG_on /*!READ_sBIT*/
#endif
#ifndef PNG_READ_INTERLACING_SUPPORTED
#   undef PNG_on /*!READ_INTERLACING*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_READ_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_READ
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_READ_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_READ_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_READ_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_READ_SUPPORTED*/" 
#endif

/* option: TRANSFORM_MECH disabled
 *   requires:  
 *   if:        
 *   enabled-by: MNG_FEATURES READ_QUANTIZE READ_GAMMA READ_BACKGROUND READ_BGR READ_EXPAND_16 READ_EXPAND READ_FILLER READ_GRAY_TO_RGB READ_INVERT_ALPHA READ_INVERT READ_PACK READ_PACKSWAP READ_RGB_TO_GRAY READ_SCALE_16_TO_8 READ_SHIFT READ_STRIP_16_TO_8 READ_STRIP_ALPHA READ_SWAP_ALPHA READ_SWAP READ_USER_TRANSFORM WRITE_SHIFT WRITE_PACK WRITE_BGR WRITE_SWAP WRITE_PACKSWAP WRITE_INVERT WRITE_FILLER WRITE_SWAP_ALPHA WRITE_USER_TRANSFORM PALETTE_MAX
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by MNG_FEATURES READ_QUANTIZE READ_GAMMA READ_BACKGROUND READ_BGR READ_EXPAND_16 READ_EXPAND READ_FILLER READ_GRAY_TO_RGB READ_INVERT_ALPHA READ_INVERT READ_PACK READ_PACKSWAP READ_RGB_TO_GRAY READ_SCALE_16_TO_8 READ_SHIFT READ_STRIP_16_TO_8 READ_STRIP_ALPHA READ_SWAP_ALPHA READ_SWAP READ_USER_TRANSFORM WRITE_SHIFT WRITE_PACK WRITE_BGR WRITE_SWAP WRITE_PACKSWAP WRITE_INVERT WRITE_FILLER WRITE_SWAP_ALPHA WRITE_USER_TRANSFORM PALETTE_MAX */
#ifdef PNG_MNG_FEATURES_SUPPORTED
#   undef PNG_not_enabled /*MNG_FEATURES*/
#endif
#ifdef PNG_READ_QUANTIZE_SUPPORTED
#   undef PNG_not_enabled /*READ_QUANTIZE*/
#endif
#ifdef PNG_READ_GAMMA_SUPPORTED
#   undef PNG_not_enabled /*READ_GAMMA*/
#endif
#ifdef PNG_READ_BACKGROUND_SUPPORTED
#   undef PNG_not_enabled /*READ_BACKGROUND*/
#endif
#ifdef PNG_READ_BGR_SUPPORTED
#   undef PNG_not_enabled /*READ_BGR*/
#endif
#ifdef PNG_READ_EXPAND_16_SUPPORTED
#   undef PNG_not_enabled /*READ_EXPAND_16*/
#endif
#ifdef PNG_READ_EXPAND_SUPPORTED
#   undef PNG_not_enabled /*READ_EXPAND*/
#endif
#ifdef PNG_READ_FILLER_SUPPORTED
#   undef PNG_not_enabled /*READ_FILLER*/
#endif
#ifdef PNG_READ_GRAY_TO_RGB_SUPPORTED
#   undef PNG_not_enabled /*READ_GRAY_TO_RGB*/
#endif
#ifdef PNG_READ_INVERT_ALPHA_SUPPORTED
#   undef PNG_not_enabled /*READ_INVERT_ALPHA*/
#endif
#ifdef PNG_READ_INVERT_SUPPORTED
#   undef PNG_not_enabled /*READ_INVERT*/
#endif
#ifdef PNG_READ_PACK_SUPPORTED
#   undef PNG_not_enabled /*READ_PACK*/
#endif
#ifdef PNG_READ_PACKSWAP_SUPPORTED
#   undef PNG_not_enabled /*READ_PACKSWAP*/
#endif
#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
#   undef PNG_not_enabled /*READ_RGB_TO_GRAY*/
#endif
#ifdef PNG_READ_SCALE_16_TO_8_SUPPORTED
#   undef PNG_not_enabled /*READ_SCALE_16_TO_8*/
#endif
#ifdef PNG_READ_SHIFT_SUPPORTED
#   undef PNG_not_enabled /*READ_SHIFT*/
#endif
#ifdef PNG_READ_STRIP_16_TO_8_SUPPORTED
#   undef PNG_not_enabled /*READ_STRIP_16_TO_8*/
#endif
#ifdef PNG_READ_STRIP_ALPHA_SUPPORTED
#   undef PNG_not_enabled /*READ_STRIP_ALPHA*/
#endif
#ifdef PNG_READ_SWAP_ALPHA_SUPPORTED
#   undef PNG_not_enabled /*READ_SWAP_ALPHA*/
#endif
#ifdef PNG_READ_SWAP_SUPPORTED
#   undef PNG_not_enabled /*READ_SWAP*/
#endif
#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
#   undef PNG_not_enabled /*READ_USER_TRANSFORM*/
#endif
#ifdef PNG_WRITE_SHIFT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_SHIFT*/
#endif
#ifdef PNG_WRITE_PACK_SUPPORTED
#   undef PNG_not_enabled /*WRITE_PACK*/
#endif
#ifdef PNG_WRITE_BGR_SUPPORTED
#   undef PNG_not_enabled /*WRITE_BGR*/
#endif
#ifdef PNG_WRITE_SWAP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_SWAP*/
#endif
#ifdef PNG_WRITE_PACKSWAP_SUPPORTED
#   undef PNG_not_enabled /*WRITE_PACKSWAP*/
#endif
#ifdef PNG_WRITE_INVERT_SUPPORTED
#   undef PNG_not_enabled /*WRITE_INVERT*/
#endif
#ifdef PNG_WRITE_FILLER_SUPPORTED
#   undef PNG_not_enabled /*WRITE_FILLER*/
#endif
#ifdef PNG_WRITE_SWAP_ALPHA_SUPPORTED
#   undef PNG_not_enabled /*WRITE_SWAP_ALPHA*/
#endif
#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
#   undef PNG_not_enabled /*WRITE_USER_TRANSFORM*/
#endif
#ifdef PNG_PALETTE_MAX_SUPPORTED
#   undef PNG_not_enabled /*PALETTE_MAX*/
#endif
#   ifndef PNG_TRANSFORM_MECH_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_TRANSFORM_MECH_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_TRANSFORM_MECH_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_TRANSFORM_MECH_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_READ_AFIRST enabled
 *   requires:   SIMPLIFIED_READ READ_SWAP_ALPHA
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SIMPLIFIED_READ_SUPPORTED
#   undef PNG_on /*!SIMPLIFIED_READ*/
#endif
#ifndef PNG_READ_SWAP_ALPHA_SUPPORTED
#   undef PNG_on /*!READ_SWAP_ALPHA*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_READ_AFIRST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_READ_AFIRST
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_READ_AFIRST_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_READ_AFIRST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_READ_AFIRST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_READ_AFIRST_SUPPORTED*/" 
#endif

/* option: SIMPLIFIED_READ_BGR enabled
 *   requires:   SIMPLIFIED_READ READ_BGR
 *   if:        
 *   enabled-by:
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#ifndef PNG_SIMPLIFIED_READ_SUPPORTED
#   undef PNG_on /*!SIMPLIFIED_READ*/
#endif
#ifndef PNG_READ_BGR_SUPPORTED
#   undef PNG_on /*!READ_BGR*/
#endif
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by */
#   ifndef PNG_SIMPLIFIED_READ_BGR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      ifdef PNG_NO_SIMPLIFIED_READ_BGR
#       undef PNG_on /*turned off*/
#      endif
#      ifdef PNG_NO_SIMPLIFIED_READ_BGR_SUPPORTED
#       undef PNG_on /*turned off*/
#      endif
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_SIMPLIFIED_READ_BGR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_SIMPLIFIED_READ_BGR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_SIMPLIFIED_READ_BGR_SUPPORTED*/" 
#endif

/* option: FORMAT_AFIRST disabled
 *   requires:  
 *   if:        
 *   enabled-by: SIMPLIFIED_READ_AFIRST SIMPLIFIED_WRITE_AFIRST
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by SIMPLIFIED_READ_AFIRST SIMPLIFIED_WRITE_AFIRST */
#ifdef PNG_SIMPLIFIED_READ_AFIRST_SUPPORTED
#   undef PNG_not_enabled /*SIMPLIFIED_READ_AFIRST*/
#endif
#ifdef PNG_SIMPLIFIED_WRITE_AFIRST_SUPPORTED
#   undef PNG_not_enabled /*SIMPLIFIED_WRITE_AFIRST*/
#endif
#   ifndef PNG_FORMAT_AFIRST_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_FORMAT_AFIRST_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_FORMAT_AFIRST_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_FORMAT_AFIRST_SUPPORTED*/" 
#endif

/* option: FORMAT_BGR disabled
 *   requires:  
 *   if:        
 *   enabled-by: SIMPLIFIED_READ_BGR SIMPLIFIED_WRITE_BGR
 *   sets:       */
#undef PNG_on
#define PNG_on 1
#undef PNG_no_if
#ifdef PNG_on /*requires, if*/
#   undef PNG_not_enabled
#   define PNG_not_enabled 1
   /* enabled by SIMPLIFIED_READ_BGR SIMPLIFIED_WRITE_BGR */
#ifdef PNG_SIMPLIFIED_READ_BGR_SUPPORTED
#   undef PNG_not_enabled /*SIMPLIFIED_READ_BGR*/
#endif
#ifdef PNG_SIMPLIFIED_WRITE_BGR_SUPPORTED
#   undef PNG_not_enabled /*SIMPLIFIED_WRITE_BGR*/
#endif
#   ifndef PNG_FORMAT_BGR_SUPPORTED /*!command line*/
#    ifdef PNG_not_enabled /*!enabled*/
#      undef PNG_on /*default off*/
#    endif /*!enabled*/
#    ifdef PNG_on
#      define PNG_FORMAT_BGR_SUPPORTED
#    endif
#   endif /*!command line*/
#   ifdef PNG_on
 PNG_DFN "#define PNG_FORMAT_BGR_SUPPORTED" 
#   endif /* definition */
#endif /*requires, if*/
#ifndef  PNG_on
 PNG_DFN "/*#undef PNG_FORMAT_BGR_SUPPORTED*/" 
#endif
PNG_DFN_END_SORT
 PNG_DFN "/* end of options */" 

/* SETTINGS */
 PNG_DFN "/* settings */" 
PNG_DFN_START_SORT 2

/* setting:  QUANTIZE_GREEN_BITS
 *   requires:
 *   default:   5 5 */
#ifdef PNG_QUANTIZE_GREEN_BITS
 PNG_DFN "#define PNG_QUANTIZE_GREEN_BITS @" PNG_QUANTIZE_GREEN_BITS "@" 
#else /* use default */
# ifdef PNG_set_QUANTIZE_GREEN_BITS
 PNG_DFN "#define PNG_QUANTIZE_GREEN_BITS @" PNG_set_QUANTIZE_GREEN_BITS "@" 
#  define PNG_QUANTIZE_GREEN_BITS 1
# else /*default*/
 PNG_DFN "#define PNG_QUANTIZE_GREEN_BITS 5" 
#  define PNG_QUANTIZE_GREEN_BITS 1
# endif /* defaults */
#endif /* setting QUANTIZE_GREEN_BITS */

/* setting:  QUANTIZE_RED_BITS
 *   requires:
 *   default:   5 5 */
#ifdef PNG_QUANTIZE_RED_BITS
 PNG_DFN "#define PNG_QUANTIZE_RED_BITS @" PNG_QUANTIZE_RED_BITS "@" 
#else /* use default */
# ifdef PNG_set_QUANTIZE_RED_BITS
 PNG_DFN "#define PNG_QUANTIZE_RED_BITS @" PNG_set_QUANTIZE_RED_BITS "@" 
#  define PNG_QUANTIZE_RED_BITS 1
# else /*default*/
 PNG_DFN "#define PNG_QUANTIZE_RED_BITS 5" 
#  define PNG_QUANTIZE_RED_BITS 1
# endif /* defaults */
#endif /* setting QUANTIZE_RED_BITS */

/* setting:  DEFAULT_GAMMA_ACCURACY
 *   requires:
 *   default:   665 665 */
#ifdef PNG_DEFAULT_GAMMA_ACCURACY
 PNG_DFN "#define PNG_DEFAULT_GAMMA_ACCURACY @" PNG_DEFAULT_GAMMA_ACCURACY "@" 
#else /* use default */
# ifdef PNG_set_DEFAULT_GAMMA_ACCURACY
 PNG_DFN "#define PNG_DEFAULT_GAMMA_ACCURACY @" PNG_set_DEFAULT_GAMMA_ACCURACY "@" 
#  define PNG_DEFAULT_GAMMA_ACCURACY 1
# else /*default*/
 PNG_DFN "#define PNG_DEFAULT_GAMMA_ACCURACY 665" 
#  define PNG_DEFAULT_GAMMA_ACCURACY 1
# endif /* defaults */
#endif /* setting DEFAULT_GAMMA_ACCURACY */

/* setting:  USER_CHUNK_CACHE_MAX
 *   requires:
 *   default:   1000 1000 */
#ifdef PNG_USER_CHUNK_CACHE_MAX
 PNG_DFN "#define PNG_USER_CHUNK_CACHE_MAX @" PNG_USER_CHUNK_CACHE_MAX "@" 
#else /* use default */
# ifdef PNG_set_USER_CHUNK_CACHE_MAX
 PNG_DFN "#define PNG_USER_CHUNK_CACHE_MAX @" PNG_set_USER_CHUNK_CACHE_MAX "@" 
#  define PNG_USER_CHUNK_CACHE_MAX 1
# else /*default*/
 PNG_DFN "#define PNG_USER_CHUNK_CACHE_MAX 1000" 
#  define PNG_USER_CHUNK_CACHE_MAX 1
# endif /* defaults */
#endif /* setting USER_CHUNK_CACHE_MAX */

/* setting:  USER_DLLFNAME_POSTFIX
 *   requires:
 *   default:   */
#ifdef PNG_USER_DLLFNAME_POSTFIX
 PNG_DFN "#define PNG_USER_DLLFNAME_POSTFIX @" PNG_USER_DLLFNAME_POSTFIX "@" 
#else /* use default */
# ifdef PNG_set_USER_DLLFNAME_POSTFIX
 PNG_DFN "#define PNG_USER_DLLFNAME_POSTFIX @" PNG_set_USER_DLLFNAME_POSTFIX "@" 
#  define PNG_USER_DLLFNAME_POSTFIX 1
# endif /* defaults */
#endif /* setting USER_DLLFNAME_POSTFIX */

/* setting:  PREFIX
 *   requires:
 *   default:   */
#ifdef PNG_PREFIX
 PNG_DFN "#define PNG_PREFIX @" PNG_PREFIX "@" 
#else /* use default */
# ifdef PNG_set_PREFIX
 PNG_DFN "#define PNG_PREFIX @" PNG_set_PREFIX "@" 
#  define PNG_PREFIX 1
# endif /* defaults */
#endif /* setting PREFIX */

/* setting:  GAMMA_THRESHOLD_FIXED
 *   requires:
 *   default:   153 153 */
#ifdef PNG_GAMMA_THRESHOLD_FIXED
 PNG_DFN "#define PNG_GAMMA_THRESHOLD_FIXED @" PNG_GAMMA_THRESHOLD_FIXED "@" 
#else /* use default */
# ifdef PNG_set_GAMMA_THRESHOLD_FIXED
 PNG_DFN "#define PNG_GAMMA_THRESHOLD_FIXED @" PNG_set_GAMMA_THRESHOLD_FIXED "@" 
#  define PNG_GAMMA_THRESHOLD_FIXED 1
# else /*default*/
 PNG_DFN "#define PNG_GAMMA_THRESHOLD_FIXED 153" 
#  define PNG_GAMMA_THRESHOLD_FIXED 1
# endif /* defaults */
#endif /* setting GAMMA_THRESHOLD_FIXED */

/* setting:  DEFAULT_READ_MACROS
 *   requires:
 *   default:   1 1 */
#ifdef PNG_DEFAULT_READ_MACROS
 PNG_DFN "#define PNG_DEFAULT_READ_MACROS @" PNG_DEFAULT_READ_MACROS "@" 
#else /* use default */
# ifdef PNG_set_DEFAULT_READ_MACROS
 PNG_DFN "#define PNG_DEFAULT_READ_MACROS @" PNG_set_DEFAULT_READ_MACROS "@" 
#  define PNG_DEFAULT_READ_MACROS 1
# else /*default*/
 PNG_DFN "#define PNG_DEFAULT_READ_MACROS 1" 
#  define PNG_DEFAULT_READ_MACROS 1
# endif /* defaults */
#endif /* setting DEFAULT_READ_MACROS */

/* setting:  QUANTIZE_BLUE_BITS
 *   requires:
 *   default:   5 5 */
#ifdef PNG_QUANTIZE_BLUE_BITS
 PNG_DFN "#define PNG_QUANTIZE_BLUE_BITS @" PNG_QUANTIZE_BLUE_BITS "@" 
#else /* use default */
# ifdef PNG_set_QUANTIZE_BLUE_BITS
 PNG_DFN "#define PNG_QUANTIZE_BLUE_BITS @" PNG_set_QUANTIZE_BLUE_BITS "@" 
#  define PNG_QUANTIZE_BLUE_BITS 1
# else /*default*/
 PNG_DFN "#define PNG_QUANTIZE_BLUE_BITS 5" 
#  define PNG_QUANTIZE_BLUE_BITS 1
# endif /* defaults */
#endif /* setting QUANTIZE_BLUE_BITS */

/* setting:  ZBUF_SIZE
 *   requires:
 *   default:   8192 8192 */
#ifdef PNG_ZBUF_SIZE
 PNG_DFN "#define PNG_ZBUF_SIZE @" PNG_ZBUF_SIZE "@" 
#else /* use default */
# ifdef PNG_set_ZBUF_SIZE
 PNG_DFN "#define PNG_ZBUF_SIZE @" PNG_set_ZBUF_SIZE "@" 
#  define PNG_ZBUF_SIZE 1
# else /*default*/
 PNG_DFN "#define PNG_ZBUF_SIZE 8192" 
#  define PNG_ZBUF_SIZE 1
# endif /* defaults */
#endif /* setting ZBUF_SIZE */

/* setting:  ABORT
 *   requires:
 *   default:   { (abort()); } { (abort()); } */
#ifdef PNG_ABORT
 PNG_DFN "#define PNG_ABORT @" PNG_ABORT "@" 
#else /* use default */
# ifdef PNG_set_ABORT
 PNG_DFN "#define PNG_ABORT @" PNG_set_ABORT "@" 
#  define PNG_ABORT 1
# else /*default*/
 PNG_DFN "#define PNG_ABORT { (abort()); }" 
#  define PNG_ABORT 1
# endif /* defaults */
#endif /* setting ABORT */

/* setting:  DEFAULT_COMPRESSION_LEVEL
 *   requires:
 *   default:   PNG_COMPRESSION_MEDIUM PNG_COMPRESSION_MEDIUM */
#ifdef PNG_DEFAULT_COMPRESSION_LEVEL
 PNG_DFN "#define PNG_DEFAULT_COMPRESSION_LEVEL @" PNG_DEFAULT_COMPRESSION_LEVEL "@" 
#else /* use default */
# ifdef PNG_set_DEFAULT_COMPRESSION_LEVEL
 PNG_DFN "#define PNG_DEFAULT_COMPRESSION_LEVEL @" PNG_set_DEFAULT_COMPRESSION_LEVEL "@" 
#  define PNG_DEFAULT_COMPRESSION_LEVEL 1
# else /*default*/
 PNG_DFN "#define PNG_DEFAULT_COMPRESSION_LEVEL PNG_COMPRESSION_MEDIUM" 
#  define PNG_DEFAULT_COMPRESSION_LEVEL 1
# endif /* defaults */
#endif /* setting DEFAULT_COMPRESSION_LEVEL */

/* setting:  USER_VERSIONINFO_COMPANYNAME
 *   requires:
 *   default:   */
#ifdef PNG_USER_VERSIONINFO_COMPANYNAME
 PNG_DFN "#define PNG_USER_VERSIONINFO_COMPANYNAME @" PNG_USER_VERSIONINFO_COMPANYNAME "@" 
#else /* use default */
# ifdef PNG_set_USER_VERSIONINFO_COMPANYNAME
 PNG_DFN "#define PNG_USER_VERSIONINFO_COMPANYNAME @" PNG_set_USER_VERSIONINFO_COMPANYNAME "@" 
#  define PNG_USER_VERSIONINFO_COMPANYNAME 1
# endif /* defaults */
#endif /* setting USER_VERSIONINFO_COMPANYNAME */

/* setting:  MAX_GAMMA_8
 *   requires:
 *   default:   11 11 */
#ifdef PNG_MAX_GAMMA_8
 PNG_DFN "#define PNG_MAX_GAMMA_8 @" PNG_MAX_GAMMA_8 "@" 
#else /* use default */
# ifdef PNG_set_MAX_GAMMA_8
 PNG_DFN "#define PNG_MAX_GAMMA_8 @" PNG_set_MAX_GAMMA_8 "@" 
#  define PNG_MAX_GAMMA_8 1
# else /*default*/
 PNG_DFN "#define PNG_MAX_GAMMA_8 11" 
#  define PNG_MAX_GAMMA_8 1
# endif /* defaults */
#endif /* setting MAX_GAMMA_8 */

/* setting:  USER_VERSIONINFO_COMMENTS
 *   requires:
 *   default:   */
#ifdef PNG_USER_VERSIONINFO_COMMENTS
 PNG_DFN "#define PNG_USER_VERSIONINFO_COMMENTS @" PNG_USER_VERSIONINFO_COMMENTS "@" 
#else /* use default */
# ifdef PNG_set_USER_VERSIONINFO_COMMENTS
 PNG_DFN "#define PNG_USER_VERSIONINFO_COMMENTS @" PNG_set_USER_VERSIONINFO_COMMENTS "@" 
#  define PNG_USER_VERSIONINFO_COMMENTS 1
# endif /* defaults */
#endif /* setting USER_VERSIONINFO_COMMENTS */

/* setting:  sRGB_PROFILE_CHECKS
 *   requires:
 *   default:   2 2 */
#ifdef PNG_sRGB_PROFILE_CHECKS
 PNG_DFN "#define PNG_sRGB_PROFILE_CHECKS @" PNG_sRGB_PROFILE_CHECKS "@" 
#else /* use default */
# ifdef PNG_set_sRGB_PROFILE_CHECKS
 PNG_DFN "#define PNG_sRGB_PROFILE_CHECKS @" PNG_set_sRGB_PROFILE_CHECKS "@" 
#  define PNG_sRGB_PROFILE_CHECKS 1
# else /*default*/
 PNG_DFN "#define PNG_sRGB_PROFILE_CHECKS 2" 
#  define PNG_sRGB_PROFILE_CHECKS 1
# endif /* defaults */
#endif /* setting sRGB_PROFILE_CHECKS */

/* setting:  ZLIB_HEADER
 *   requires:
 *   default:   <zlib.h> <zlib.h> */
#ifdef PNG_ZLIB_HEADER
 PNG_DFN "#define PNG_ZLIB_HEADER @" PNG_ZLIB_HEADER "@" 
#else /* use default */
# ifdef PNG_set_ZLIB_HEADER
 PNG_DFN "#define PNG_ZLIB_HEADER @" PNG_set_ZLIB_HEADER "@" 
#  define PNG_ZLIB_HEADER 1
# else /*default*/
 PNG_DFN "#define PNG_ZLIB_HEADER <zlib.h>" 
#  define PNG_ZLIB_HEADER 1
# endif /* defaults */
#endif /* setting ZLIB_HEADER */

/* setting:  sCAL_PRECISION
 *   requires:
 *   default:   5 5 */
#ifdef PNG_sCAL_PRECISION
 PNG_DFN "#define PNG_sCAL_PRECISION @" PNG_sCAL_PRECISION "@" 
#else /* use default */
# ifdef PNG_set_sCAL_PRECISION
 PNG_DFN "#define PNG_sCAL_PRECISION @" PNG_set_sCAL_PRECISION "@" 
#  define PNG_sCAL_PRECISION 1
# else /*default*/
 PNG_DFN "#define PNG_sCAL_PRECISION 5" 
#  define PNG_sCAL_PRECISION 1
# endif /* defaults */
#endif /* setting sCAL_PRECISION */

/* setting:  USER_BUG_REPORTS
 *   requires:
 *   default:   */
#ifdef PNG_USER_BUG_REPORTS
 PNG_DFN "#define PNG_USER_BUG_REPORTS @" PNG_USER_BUG_REPORTS "@" 
#else /* use default */
# ifdef PNG_set_USER_BUG_REPORTS
 PNG_DFN "#define PNG_USER_BUG_REPORTS @" PNG_set_USER_BUG_REPORTS "@" 
#  define PNG_USER_BUG_REPORTS 1
# endif /* defaults */
#endif /* setting USER_BUG_REPORTS */

/* setting:  USER_HEIGHT_MAX
 *   requires:
 *   default:   1000000 1000000 */
#ifdef PNG_USER_HEIGHT_MAX
 PNG_DFN "#define PNG_USER_HEIGHT_MAX @" PNG_USER_HEIGHT_MAX "@" 
#else /* use default */
# ifdef PNG_set_USER_HEIGHT_MAX
 PNG_DFN "#define PNG_USER_HEIGHT_MAX @" PNG_set_USER_HEIGHT_MAX "@" 
#  define PNG_USER_HEIGHT_MAX 1
# else /*default*/
 PNG_DFN "#define PNG_USER_HEIGHT_MAX 1000000" 
#  define PNG_USER_HEIGHT_MAX 1
# endif /* defaults */
#endif /* setting USER_HEIGHT_MAX */

/* setting:  INFLATE_BUF_SIZE
 *   requires:
 *   default:   1024 1024 */
#ifdef PNG_INFLATE_BUF_SIZE
 PNG_DFN "#define PNG_INFLATE_BUF_SIZE @" PNG_INFLATE_BUF_SIZE "@" 
#else /* use default */
# ifdef PNG_set_INFLATE_BUF_SIZE
 PNG_DFN "#define PNG_INFLATE_BUF_SIZE @" PNG_set_INFLATE_BUF_SIZE "@" 
#  define PNG_INFLATE_BUF_SIZE 1
# else /*default*/
 PNG_DFN "#define PNG_INFLATE_BUF_SIZE 1024" 
#  define PNG_INFLATE_BUF_SIZE 1
# endif /* defaults */
#endif /* setting INFLATE_BUF_SIZE */

/* setting:  USER_PRIVATEBUILD
 *   requires:
 *   default:   */
#ifdef PNG_USER_PRIVATEBUILD
 PNG_DFN "#define PNG_USER_PRIVATEBUILD @" PNG_USER_PRIVATEBUILD "@" 
#else /* use default */
# ifdef PNG_set_USER_PRIVATEBUILD
 PNG_DFN "#define PNG_USER_PRIVATEBUILD @" PNG_set_USER_PRIVATEBUILD "@" 
#  define PNG_USER_PRIVATEBUILD 1
# endif /* defaults */
#endif /* setting USER_PRIVATEBUILD */

/* setting:  USER_CHUNK_MALLOC_MAX
 *   requires:
 *   default:   8000000 8000000 */
#ifdef PNG_USER_CHUNK_MALLOC_MAX
 PNG_DFN "#define PNG_USER_CHUNK_MALLOC_MAX @" PNG_USER_CHUNK_MALLOC_MAX "@" 
#else /* use default */
# ifdef PNG_set_USER_CHUNK_MALLOC_MAX
 PNG_DFN "#define PNG_USER_CHUNK_MALLOC_MAX @" PNG_set_USER_CHUNK_MALLOC_MAX "@" 
#  define PNG_USER_CHUNK_MALLOC_MAX 1
# else /*default*/
 PNG_DFN "#define PNG_USER_CHUNK_MALLOC_MAX 8000000" 
#  define PNG_USER_CHUNK_MALLOC_MAX 1
# endif /* defaults */
#endif /* setting USER_CHUNK_MALLOC_MAX */

/* setting:  ZLIB_VERNUM
 *   requires:
 *   default:   @ZLIB_VERNUM @" ZLIB_VERNUM "@ */
#ifdef PNG_ZLIB_VERNUM
 PNG_DFN "#define PNG_ZLIB_VERNUM @" PNG_ZLIB_VERNUM "@" 
#else /* use default */
# ifdef PNG_set_ZLIB_VERNUM
 PNG_DFN "#define PNG_ZLIB_VERNUM @" PNG_set_ZLIB_VERNUM "@" 
#  define PNG_ZLIB_VERNUM 1
# else /*default*/
 PNG_DFN "#define PNG_ZLIB_VERNUM @" ZLIB_VERNUM "@" 
#  define PNG_ZLIB_VERNUM 1
# endif /* defaults */
#endif /* setting ZLIB_VERNUM */

/* setting:  USER_WIDTH_MAX
 *   requires:
 *   default:   1000000 1000000 */
#ifdef PNG_USER_WIDTH_MAX
 PNG_DFN "#define PNG_USER_WIDTH_MAX @" PNG_USER_WIDTH_MAX "@" 
#else /* use default */
# ifdef PNG_set_USER_WIDTH_MAX
 PNG_DFN "#define PNG_USER_WIDTH_MAX @" PNG_set_USER_WIDTH_MAX "@" 
#  define PNG_USER_WIDTH_MAX 1
# else /*default*/
 PNG_DFN "#define PNG_USER_WIDTH_MAX 1000000" 
#  define PNG_USER_WIDTH_MAX 1
# endif /* defaults */
#endif /* setting USER_WIDTH_MAX */

/* setting:  ZLIB_PREFIX
 *   requires:
 *   default:   */
#ifdef PNG_ZLIB_PREFIX
 PNG_DFN "#define PNG_ZLIB_PREFIX @" PNG_ZLIB_PREFIX "@" 
#else /* use default */
# ifdef PNG_set_ZLIB_PREFIX
 PNG_DFN "#define PNG_ZLIB_PREFIX @" PNG_set_ZLIB_PREFIX "@" 
#  define PNG_ZLIB_PREFIX 1
# endif /* defaults */
#endif /* setting ZLIB_PREFIX */

/* setting:  USER_CONFIG
 *   requires:
 *   default:   */
#ifdef PNG_USER_CONFIG
 PNG_DFN "#define PNG_USER_CONFIG @" PNG_USER_CONFIG "@" 
#else /* use default */
# ifdef PNG_set_USER_CONFIG
 PNG_DFN "#define PNG_USER_CONFIG @" PNG_set_USER_CONFIG "@" 
#  define PNG_USER_CONFIG 1
# endif /* defaults */
#endif /* setting USER_CONFIG */

/* setting:  EXTENSION_HEADER
 *   requires:
 *   default:   */
#ifdef PNG_EXTENSION_HEADER
 PNG_DFN "#define PNG_EXTENSION_HEADER @" PNG_EXTENSION_HEADER "@" 
#else /* use default */
# ifdef PNG_set_EXTENSION_HEADER
 PNG_DFN "#define PNG_EXTENSION_HEADER @" PNG_set_EXTENSION_HEADER "@" 
#  define PNG_EXTENSION_HEADER 1
# endif /* defaults */
#endif /* setting EXTENSION_HEADER */

/* setting:  API_RULE
 *   requires:
 *   default:   0 0 */
#ifdef PNG_API_RULE
 PNG_DFN "#define PNG_API_RULE @" PNG_API_RULE "@" 
#else /* use default */
# ifdef PNG_set_API_RULE
 PNG_DFN "#define PNG_API_RULE @" PNG_set_API_RULE "@" 
#  define PNG_API_RULE 1
# else /*default*/
 PNG_DFN "#define PNG_API_RULE 0" 
#  define PNG_API_RULE 1
# endif /* defaults */
#endif /* setting API_RULE */

/* setting:  IDAT_READ_SIZE
 *   requires:
 *   default:   4096 4096 */
#ifdef PNG_IDAT_READ_SIZE
 PNG_DFN "#define PNG_IDAT_READ_SIZE @" PNG_IDAT_READ_SIZE "@" 
#else /* use default */
# ifdef PNG_set_IDAT_READ_SIZE
 PNG_DFN "#define PNG_IDAT_READ_SIZE @" PNG_set_IDAT_READ_SIZE "@" 
#  define PNG_IDAT_READ_SIZE 1
# else /*default*/
 PNG_DFN "#define PNG_IDAT_READ_SIZE 4096" 
#  define PNG_IDAT_READ_SIZE 1
# endif /* defaults */
#endif /* setting IDAT_READ_SIZE */

/* setting:  USER_VERSIONINFO_LEGALTRADEMARKS
 *   requires:
 *   default:   */
#ifdef PNG_USER_VERSIONINFO_LEGALTRADEMARKS
 PNG_DFN "#define PNG_USER_VERSIONINFO_LEGALTRADEMARKS @" PNG_USER_VERSIONINFO_LEGALTRADEMARKS "@" 
#else /* use default */
# ifdef PNG_set_USER_VERSIONINFO_LEGALTRADEMARKS
 PNG_DFN "#define PNG_USER_VERSIONINFO_LEGALTRADEMARKS @" PNG_set_USER_VERSIONINFO_LEGALTRADEMARKS "@" 
#  define PNG_USER_VERSIONINFO_LEGALTRADEMARKS 1
# endif /* defaults */
#endif /* setting USER_VERSIONINFO_LEGALTRADEMARKS */
PNG_DFN_END_SORT
 PNG_DFN "/* end of settings */" 
 PNG_DFN "#endif /* PNGLCONF_H */" 
