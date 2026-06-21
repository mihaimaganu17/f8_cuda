all:
	nvcc .\vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory -Xcompiler /Zc:preprocessor -std=c++17
	nvcc .\err_test.cu -o err_test Xcompiler "/Zc:preprocessor" -std=c++1