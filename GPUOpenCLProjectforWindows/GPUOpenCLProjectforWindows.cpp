/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>

 // Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// Khronos ICD supports only latest OpenCL version
#define CL_TARGET_OPENCL_VERSION 220

// Suppress a compiler warning about 'clCreateCommandQueue': was declared deprecated
// for OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL\cl.h"
#include "utils.h"

//for perf. counters
#include <Windows.h>

/* This function helps to create informative messages in
 * case when OpenCL errors occur. It returns a string
 * representation for an OpenCL error code.
 * (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
 */
const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}


/* Convenient container for all OpenCL specific objects used in the sample
 *
 * It consists of two parts:
 *   - regular OpenCL objects which are used in almost each normal OpenCL applications
 *   - several OpenCL objects that are specific for this particular sample
 *
 * You collect all these objects in one structure for utility purposes
 * only, there is no OpenCL specific here: just to avoid global variables
 * and make passing all these arguments in functions easier.
 */
struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id     device;            // hold the selected device handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel;            // hold the kernel handler
	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)

	// Objects that are specific for algorithm implemented in this sample
	cl_mem           srcA;              // hold first source buffer
	cl_mem           srcB;              // hold second source buffer
	cl_mem           dstMem;            // hold destination buffer
};

ocl_args_d_t::ocl_args_d_t() :
	context(NULL),
	device(NULL),
	commandQueue(NULL),
	program(NULL),
	kernel(NULL),
	platformVersion(OPENCL_VERSION_1_2),
	deviceVersion(OPENCL_VERSION_1_2),
	compilerVersion(OPENCL_VERSION_1_2),
	srcA(NULL),
	srcB(NULL),
	dstMem(NULL)
{
}

/*
 * destructor - called only once
 * Release all OpenCL objects
 * This is a regular sequence of calls to deallocate all created OpenCL resources in bootstrapOpenCL.
 *
 * You may want to call these deallocation procedures in the middle of your application execution
 * (not at the end) if you don't further need OpenCL runtime.
 * You may want to do that in order to free some memory, for example,
 * or recreate OpenCL objects with different parameters.
 *
 */
ocl_args_d_t::~ocl_args_d_t()
{
	cl_int err = CL_SUCCESS;

	if (kernel)
	{
		err = clReleaseKernel(kernel);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (program)
	{
		err = clReleaseProgram(program);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcA)
	{
		err = clReleaseMemObject(srcA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcB)
	{
		err = clReleaseMemObject(srcB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMem)
	{
		err = clReleaseMemObject(dstMem);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (commandQueue)
	{
		err = clReleaseCommandQueue(commandQueue);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (device)
	{
		err = clReleaseDevice(device);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (context)
	{
		err = clReleaseContext(context);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	/*
	 * Note there is no procedure to deallocate platform
	 * because it was not created at the startup,
	 * but just queried from OpenCL runtime.
	 */
}


/*
 * Check whether an OpenCL platform is the required platform
 * (based on the platform's name)
 */
bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;
	bool match = false;

	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
	// The value returned in stringLength
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now, that we know the platform's name string length, we can allocate enough space before read it
	std::vector<char> platformName(stringLength);

	// Read the platform's name string
	// The read value returned in platformName
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now check if the platform's name is the required one
	if (strstr(&platformName[0], preferredPlatform) != 0)
	{
		// The checked platform is the one we're looking for
		LogInfo("Platform: %s\n", &platformName[0]);
		match = true;
	}

	return match;
}

/*
 * Find and return the preferred OpenCL platform
 * In case that preferredPlatform is NULL, the ID of the first discovered platform will be returned
 */
cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
	cl_uint numPlatforms = 0;
	cl_int err = CL_SUCCESS;

	// Get (in numPlatforms) the number of OpenCL platforms available
	// No platform ID will be return, since platforms is NULL
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}
	LogInfo("Number of available platforms: %u\n", numPlatforms);

	if (0 == numPlatforms)
	{
		LogError("Error: No platforms found!\n");
		return NULL;
	}

	std::vector<cl_platform_id> platforms(numPlatforms);

	// Now, obtains a list of numPlatforms OpenCL platforms available
	// The list of platforms available will be returned in platforms
	err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}

	// Check if one of the available platform matches the preferred requirements
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		bool match = true;
		cl_uint numDevices = 0;

		// If the preferredPlatform is not NULL then check if platforms[i] is the required one
		// Otherwise, continue the check with platforms[i]
		if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
		{
			// In case we're looking for a specific platform
			match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
		}

		// match is true if the platform's name is the required one or don't care (NULL)
		if (match)
		{
			// Obtains the number of deviceType devices available on platform
			// When the function failed we expect numDevices to be zero.
			// We ignore the function return value since a non-zero error code
			// could happen if this platform doesn't support the specified device type.
			err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
			if (CL_SUCCESS != err)
			{
				LogInfo("   Required device was not found on this platform.\n");
			}

			if (0 != numDevices)
			{
				// There is at list one device that answer the requirements
				LogInfo("   Required device was found.\n");
				return platforms[i];
			}
		}
	}

	LogError("Error: Required device was not found on any platform.\n");
	return NULL;
}


/*
 * This function read the OpenCL platdorm and device versions
 * (using clGetxxxInfo API) and stores it in the ocl structure.
 * Later it will enable us to support both OpenCL 1.2 and 2.0 platforms and devices
 * in the same program.
 */
int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t* ocl)
{
	cl_int err = CL_SUCCESS;

	// Read the platform's version string length (param_value is NULL).
	// The value returned in stringLength
	size_t stringLength = 0;
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the platform's version string length, we can allocate enough space before read it
	std::vector<char> platformVersion(stringLength);

	// Read the platform's version string
	// The read value returned in platformVersion
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->platformVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's version string length, we can allocate enough space before read it
	std::vector<char> deviceVersion(stringLength);

	// Read the device's version string
	// The read value returned in deviceVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->deviceVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's OpenCL C version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
	std::vector<char> compilerVersion(stringLength);

	// Read the device's OpenCL C version string
	// The read value returned in compilerVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
	{
		ocl->compilerVersion = OPENCL_VERSION_2_0;
	}

	return err;
}


/*
 * Generate random value for input buffers
 */
void generateInput(cl_int* inputArray, cl_uint arrayWidth, cl_uint arrayHeight)
{
	srand(12345);

	// random initialization of input
	cl_uint array_size = arrayWidth * arrayHeight;
	for (cl_uint i = 0; i < array_size; ++i)
	{
		inputArray[i] = rand();
	}
}


/*
 * This function picks/creates necessary OpenCL objects which are needed.
 * The objects are:
 * OpenCL platform, device, context, and command queue.
 *
 * All these steps are needed to be performed once in a regular OpenCL application.
 * This happens before actual compute kernels calls are performed.
 *
 * For convenience, in this application you store all those basic OpenCL objects in structure ocl_args_d_t,
 * so this function populates fields of this structure, which is passed as parameter ocl.
 * Please, consider reviewing the fields before going further.
 * The structure definition is right in the beginning of this file.
 */
int SetupOpenCL(ocl_args_d_t* ocl, cl_device_type deviceType)
{
	// The following variable stores return codes for all OpenCL calls.
	cl_int err = CL_SUCCESS;

	// Query for all available OpenCL platforms on the system
	// Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
	cl_platform_id platformId = FindOpenCLPlatform("Intel", deviceType);
	if (NULL == platformId)
	{
		LogError("Error: Failed to find OpenCL platform.\n");
		return CL_INVALID_VALUE;
	}

	// Create context with device of specified type.
	// Required device type is passed as function argument deviceType.
	// So you may use this function to create context for any CPU or GPU OpenCL device.
	// The creation is synchronized (pfn_notify is NULL) and NULL user_data
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
	if ((CL_SUCCESS != err) || (NULL == ocl->context))
	{
		LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Query for OpenCL device which was used for context creation
	err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	// Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
	GetPlatformAndDeviceVersion(platformId, ocl);

	// Create command queue.
	// OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
	// Command queue guarantees some ordering between calls and other OpenCL commands.
	// Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0
	if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
	{
		const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
	}
	else {
		// default behavior: OpenCL 1.2
		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
		ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
	}
#else
	// default behavior: OpenCL 1.2
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}


/*
 * Create and build OpenCL program from its source code
 */
int CreateAndBuildProgram(ocl_args_d_t* ocl)
{
	cl_int err = CL_SUCCESS;

	// Upload the OpenCL C source code from the input file to source
	// The size of the C program is returned in sourceSize
	char* source = NULL;
	size_t src_size = 0;
	err = ReadSourceFromFile("Template.cl", &source, &src_size);
	if (CL_SUCCESS != err)
	{
		LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &src_size, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// Build the program
	// During creation a program is not built. You need to explicitly call build function.
	// Here you just use create-build sequence,
	// but there are also other possibilities when program consist of several parts,
	// some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
	// alternatives.
	//err = clBuildProgram(ocl->program, 1, &ocl->device, "-cl-opt-disable", NULL, NULL);
	err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

		// In case of error print the build log to the standard output
		// First check the size of the log
		// Then allocate the memory and obtain the log from the program
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			std::vector<char> build_log(log_size);
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

			LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
		}
	}

Finish:
	if (source)
	{
		delete[] source;
		source = NULL;
	}

	return err;
}


/*
 * Create OpenCL buffers from host memory
 * These buffers will be used later by the OpenCL kernel
 */
int CreateBufferArguments(ocl_args_d_t* ocl, cl_int* inputA, cl_int* inputB, cl_int* outputC, cl_uint arrayWidth, cl_uint arrayHeight)
{
	cl_int err = CL_SUCCESS;

	cl_image_format format;
	cl_image_desc desc;

	// Define the image data-type and order -
	// one channel (R) with unit values
	format.image_channel_data_type = CL_UNSIGNED_INT32;
	format.image_channel_order = CL_R;

	// Define the image properties (descriptor)
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = arrayWidth;
	desc.image_height = arrayHeight;
	desc.image_depth = 0;
	desc.image_array_size = 1;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
#ifdef CL_VERSION_2_0
	desc.mem_object = NULL;
#else
	desc.buffer = NULL;
#endif

	// Create first image based on host memory inputA
	ocl->srcA = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, inputA, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create second image based on host memory inputB
	ocl->srcB = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, inputB, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for srcB returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create third (output) image based on host memory outputC
	ocl->dstMem = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, outputC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for dstMem returned %s\n", TranslateOpenCLError(err));
		return err;
	}


	return CL_SUCCESS;
}


/*
 * Set kernel arguments
 */
cl_uint SetKernelArguments(ocl_args_d_t* ocl)
{
	cl_int err = CL_SUCCESS;

	err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void*)&ocl->srcA);
	if (CL_SUCCESS != err)
	{
		LogError("error: Failed to set argument srcA, returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void*)&ocl->srcB);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), (void*)&ocl->dstMem);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	return err;
}


/*
 * Execute the kernel
 */
cl_uint ExecuteAddKernel(ocl_args_d_t* ocl, cl_uint width, cl_uint height)
{
	cl_int err = CL_SUCCESS;

	// Define global iteration space for clEnqueueNDRangeKernel.
	size_t globalWorkSize[2] = { width, height };


	// execute kernel
	err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Wait until the queued kernel is completed by the device
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}


/*
 * "Read" the result buffer (mapping the buffer to the host memory address)
 */
bool ReadAndVerify(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_int* inputA, cl_int* inputB)
{
	cl_int err = CL_SUCCESS;
	bool result = true;

	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
	// The map operation is blocking
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { width, height, 1 };
	size_t image_row_pitch;
	size_t image_slice_pitch;
	cl_int* resultPtr = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMem, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
		return false;
	}

	// Call clFinish to guarantee that output region is updated
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
	// Verify the results
	unsigned int size = width * height;
	for (unsigned int k = 0; k < size; ++k)
	{
		if (resultPtr[k] != inputA[k] + inputB[k])
		{
			LogError("Verification failed at %d: (%d + %d = %d)\n", k, inputA[k], inputB[k], resultPtr[k]);
			result = false;
		}
	}

	// Unmapped the output buffer before releasing it
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMem, resultPtr, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}

	return result;
}

int myTest();
int myTestCPU();

/*
 * main execution routine
 * Basically it consists of three parts:
 *   - generating the inputs
 *   - running OpenCL kernel
 *   - reading results of processing
 */
int _tmain(int argc, TCHAR* argv[])
{
	int ret;
	ret = myTest();
	if (ret != 0)
		return ret;
	ret = myTestCPU();
	return ret;
}

size_t zeroCopySizeAlignment(size_t requiredSize)
{
	// Please refer to Intel Zero Copy Tutorial and OpenCL Performance Guide
	// The following statement rounds requiredSize up to the next 64-byte boundary
	return requiredSize + (~requiredSize + 1) % 64;   // or even shorter: requiredSize + (-requiredSize) % 64
}

#define ALIGNED_MALLOC_BYTES 4096

int myTest()
{
	cl_int err;
	ocl_args_d_t ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
	{
		return -1;
	}

	ocl.kernel = clCreateKernel(ocl.program, "GetValueOfB", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	const int widthL0 = 3072;
	const int heightL0 = 3072;

	// comment - (2022-05-26, Ethan) 24 * 24 is the 7th layer in Laplacian pyramid
	//const int layerIndex = 7;
	//const int widthL8 = 24;
	//const int heightL8 = 24;

	// comment - (2022-05-26, Ethan) 12 * 12 is the 8th layer in Laplacian pyramid
	const int layerIndex = 8;
	const int widthL8 = 12;
	const int heightL8 = 12;

	const float sigmaR = 1.0f;
	const float alpha = 1.0f;
	const float beta = 1.0f;

	size_t matG0_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * heightL0);
	float* pMatG0 = (float*)_aligned_malloc(matG0_size, ALIGNED_MALLOC_BYTES);
	memset(pMatG0, 0, matG0_size);

	size_t matG8_size = zeroCopySizeAlignment(sizeof(float) * widthL8 * heightL8);
	float* pMatG8 = (float*)_aligned_malloc(matG8_size, ALIGNED_MALLOC_BYTES);
	memset(pMatG8, 0, matG8_size);

	size_t matCoeffGR_size = zeroCopySizeAlignment(sizeof(float) * heightL0 * heightL8);
	float* pMatCoeffGR = (float*)_aligned_malloc(matCoeffGR_size, ALIGNED_MALLOC_BYTES);
	// comment - (2022-05-06, Ethan) In reality, pMatCoeffGR has some values other than zeros.
	// But to reproduce the phenomenon it's okay even if they are zeros and it makes analysis easier.
	// This is the same for pMatCoeffGUpR, pMatCoeffGC, pMatCoeffGUpC
	memset(pMatCoeffGR, 0, matCoeffGR_size);

	size_t matCoeffGUpR_size = zeroCopySizeAlignment(sizeof(float) * heightL0 * heightL8);
	float* pMatCoeffGUpR = (float*)_aligned_malloc(matCoeffGUpR_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGUpR, 0, matCoeffGUpR_size);

	size_t matCoeffGC_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * widthL8);
	float* pMatCoeffGC = (float*)_aligned_malloc(matCoeffGC_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGC, 0, matCoeffGC_size);

	size_t matCoeffGUpC_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * widthL8);
	float* pMatCoeffGUpC = (float*)_aligned_malloc(matCoeffGUpC_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGUpC, 0, matCoeffGUpC_size);

	size_t matB_size = zeroCopySizeAlignment(sizeof(float) * widthL8 * heightL8);
	// comment - (2022-05-26, Ethan) should all zero after OpenCL kernel
	float* pMatB = (float*)_aligned_malloc(matB_size, ALIGNED_MALLOC_BYTES);
	// comment - (2022-05-26, Ethan) initialize as nan
	memset(pMatB, 0xff, matB_size);


	cl_mem memMatG0 = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matG0_size, pMatG0, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	cl_mem memMatG8 = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matG8_size, pMatG8, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	cl_mem memCoeffGR = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matCoeffGR_size, pMatCoeffGR, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	cl_mem memCoeffGUpR = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matCoeffGUpR_size, pMatCoeffGUpR, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	cl_mem memCoeffGC = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matCoeffGC_size, pMatCoeffGC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	cl_mem memCoeffGUpC = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matCoeffGUpC_size, pMatCoeffGUpC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	cl_mem memMatB = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, matB_size, pMatB, &err); ;
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	const int halfWidthFilter = 3 * (2 << layerIndex) - 2;

	cl_uint argIndex = 0;
	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memMatG0);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &widthL0);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &heightL0);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &layerIndex);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &halfWidthFilter);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memCoeffGR);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &heightL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memCoeffGUpR);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &heightL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memCoeffGC);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &widthL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memCoeffGUpC);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &widthL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memMatB);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &widthL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_int), &heightL8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_mem), &memMatG8);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_float), &sigmaR);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_float), &alpha);
	if (CL_SUCCESS != err)
		return -1;

	err = clSetKernelArg(ocl.kernel, argIndex++, sizeof(cl_float), &beta);
	if (CL_SUCCESS != err)
		return -1;

	if (CL_SUCCESS != ExecuteAddKernel(&ocl, size_t(widthL8), size_t(heightL8)))
	{
		return -1;
	}

	err = clEnqueueReadBuffer(ocl.commandQueue, memMatB, CL_TRUE, 0, matB_size, pMatB, 0, nullptr, nullptr);

	clReleaseMemObject(memCoeffGR);
	clReleaseMemObject(memCoeffGUpR);
	clReleaseMemObject(memCoeffGC);
	clReleaseMemObject(memCoeffGUpC);
	clReleaseMemObject(memMatB);
	clReleaseMemObject(memMatG8);
	clReleaseMemObject(memMatG0);

	_aligned_free(pMatB);
	_aligned_free(pMatCoeffGUpC);
	_aligned_free(pMatCoeffGC);
	_aligned_free(pMatCoeffGUpR);
	_aligned_free(pMatCoeffGR);
	_aligned_free(pMatG8);
	_aligned_free(pMatG0);

	return 0;
}

double DetailRemap(const double alpha, double delta, double sigmaR);
double EdgeRemap(const double beta, double delta);
double SmoothStep(double xMin, double xMax, double x);

int myTestCPU()
{
	const int widthL0 = 3072;
	const int heightL0 = 3072;

	// comment - (2022-05-26, Ethan) 24 * 24 is the 7th layer in Laplacian pyramid
	//const int layerIndex = 7;
	//const int widthL8 = 24;
	//const int heightL8 = 24;

	// comment - (2022-05-26, Ethan) 12 * 12 is the 8th layer in Laplacian pyramid
	const int layerIndex = 8;
	const int widthL8 = 12;
	const int heightL8 = 12;

	const float sigmaR = 1.0f;
	const float alpha = 1.0f;
	const float beta = 1.0f;


	size_t matG0_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * heightL0);
	float* pMatG0 = (float*)_aligned_malloc(matG0_size, ALIGNED_MALLOC_BYTES);
	memset(pMatG0, 0, matG0_size);

	size_t matG8_size = zeroCopySizeAlignment(sizeof(float) * widthL8 * heightL8);
	float* pMatG8 = (float*)_aligned_malloc(matG8_size, ALIGNED_MALLOC_BYTES);
	memset(pMatG8, 0, matG8_size);

	size_t matCoeffGR_size = zeroCopySizeAlignment(sizeof(float) * heightL0 * heightL8);
	float* pMatCoeffGR = (float*)_aligned_malloc(matCoeffGR_size, ALIGNED_MALLOC_BYTES);
	// comment - (2022-05-06, Ethan) In reality, pMatCoeffGR has some values other than zeros.
	// But to reproduce the phenomenon it's okay even if they are zeros and it makes analysis easier.
	// This is the same for pMatCoeffGUpR, pMatCoeffGC, pMatCoeffGUpC
	memset(pMatCoeffGR, 0, matCoeffGR_size);

	size_t matCoeffGUpR_size = zeroCopySizeAlignment(sizeof(float) * heightL0 * heightL8);
	float* pMatCoeffGUpR = (float*)_aligned_malloc(matCoeffGUpR_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGUpR, 0, matCoeffGUpR_size);

	size_t matCoeffGC_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * widthL8);
	float* pMatCoeffGC = (float*)_aligned_malloc(matCoeffGC_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGC, 0, matCoeffGC_size);

	size_t matCoeffGUpC_size = zeroCopySizeAlignment(sizeof(float) * widthL0 * widthL8);
	float* pMatCoeffGUpC = (float*)_aligned_malloc(matCoeffGUpC_size, ALIGNED_MALLOC_BYTES);
	memset(pMatCoeffGUpC, 0, matCoeffGUpC_size);

	size_t matB_size = zeroCopySizeAlignment(sizeof(float) * widthL8 * heightL8);
	// comment - (2022-05-26, Ethan) should all zero after OpenCL kernel
	float* pMatB = (float*)_aligned_malloc(matB_size, ALIGNED_MALLOC_BYTES);
	// comment - (2022-05-26, Ethan) initialize as nan
	memset(pMatB, 0xff, matB_size);

	const int halfWidthFilter = 3 * (2 << layerIndex) - 2;

	for (int yB = 0; yB < heightL8; yB++)
	{
		for (int xB = 0; xB < widthL8; xB++)
		{
			int yG0 = yB << layerIndex;
			const int nROITop = max(yG0 - halfWidthFilter, 0);
			const int nROIBottom = min(yG0 + halfWidthFilter, heightL0 - 1);
			const int nROIHeight = nROIBottom - nROITop + 1;

			int xG0 = xB << layerIndex;
			const int nROILeft = max(xG0 - halfWidthFilter, 0);
			const int nROIRight = min(xG0 + halfWidthFilter, widthL0 - 1);
			const int nROIWidth = nROIRight - nROILeft + 1;

			const float* pfG0Local = (const float*)(pMatG0 + nROITop * widthL0 + nROILeft);

			float* pCurMatCoeffGR = pMatCoeffGR + heightL8 * yB;
			float* pCurMatCoeffGUpR = pMatCoeffGUpR + heightL8 * yB;

			float* pCurMatCoeffGC = pMatCoeffGC + widthL8 * xB;
			float* pCurMatCoeffGUpC = pMatCoeffGUpC + widthL8 * xB;

			const double reference = pMatG8[yB * widthL8 + xB]; // should be 0.0

			double dSum = 0.0;
			for (int nY = 0; nY < nROIHeight; nY++)
			{
				for (int nX = 0; nX < nROIWidth; nX++)
				{
					double value = pfG0Local[nY * widthL0 + nX]; // should be 0.0

					// comment  - (2022-05-31, Ethan) should be met
					//if (nROITop * widthL0 + nROILeft + nY * widthL0 + nX >= widthL0 * heightL0)
					//	return -1;

					double delta = fabs(value - reference); // should be 0.0
					double sign = value < reference ? -1. : 1.; // should be 1.0
					double output = 0.;
					if (delta < sigmaR) // should be true because (0.0 < 1.0) 
						// should be 0.0 = 0.0 + 1.0 * 1.0 * DetailRemap(1.0, 0.0, 1.0);
						// should be 0.0 = 0.0 + 1.0 * 1.0 * 0.0
						output = reference + sign * sigmaR * DetailRemap(alpha, delta, sigmaR);
					else
						// next line should not be called
						output = reference + sign * (EdgeRemap(beta, delta - sigmaR) + sigmaR);
					dSum += (pMatCoeffGC[nX] * pMatCoeffGR[nY] - pMatCoeffGUpC[nX] * pMatCoeffGUpR[nY]) * output;
					// should be dSum += (0.0 * 0.0 - 0.0 * 0.0) * 0.0
					// should be dSum += 0.0
				}
			}

			printf("(%d, %d, %f)\n", xB, yB, dSum);
			pMatB[yB * widthL8 + xB] = (float)dSum;
		}
	}

	_aligned_free(pMatB);
	_aligned_free(pMatCoeffGUpC);
	_aligned_free(pMatCoeffGC);
	_aligned_free(pMatCoeffGUpR);
	_aligned_free(pMatCoeffGR);
	_aligned_free(pMatG8);
	_aligned_free(pMatG0);

	return 0;
}

double DetailRemap(const double alpha, double delta, double sigmaR)
{
	double fraction = delta / sigmaR; // 0.0 / 1.0
	double polynomial = pow(fraction, alpha); // pow(0.0, 1.0)
	if (alpha < 1.0) // always false because (1.0 < 1.0)
	{
		const double kNoiseLevel = 0.01;
		double blend = SmoothStep(kNoiseLevel, 2.0 * kNoiseLevel, fraction * sigmaR);
		polynomial = blend * polynomial + (1.0 - blend) * fraction;
	}
	return polynomial; // 0.0
}

double EdgeRemap(const double beta, double delta)
{
	return beta * delta;
}

double SmoothStep(double xMin, double xMax, double x)
{
	double y = (x - xMin) / (xMax - xMin);
	y = fmax(0.0, fmin(1.0, y));
	return pow(y, 2.0) * pow(y - 2.0, 2.0);
}
