# GPUOpenCLProjectforWindows



## Compile and test environment
- Visual Studio 2019 Professional 2019
- Intel® SDK for OpenCL™ Applications 2020.3.494
- Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz
- Intel® Distribution for GDB Target **must not be installed**

## Project description
- The project was created using "GPU OpenCL Project for Windows"
![image](https://user-images.githubusercontent.com/83520888/170445979-f58de3af-ef41-4bcf-8e07-88ec285d5306.png)
- My code starts from line 803 in GPUOpenCLProjectforWindows.cpp. All the code above was created by the template.
- Template.cl includes my kernel.

## Expected printf output
There're a couple of ways to get expected result.
- Add "-cl-opt-disable" when compiling the kernel program
- Install "Intel® Distribution for GDB Target"
- Use NVIDIA instead of using HD Graphics
```
Number of available platforms: 5
Platform: Intel(R) OpenCL HD Graphics
   Required device was found.
(0, 0)
(1, 0)
(2, 0)
(3, 0)
(0, 1)
(1, 1)
(2, 1)
(3, 1)
(4, 2)
(5, 2)
(6, 2)
(7, 2)
(4, 3)
(5, 3)
(6, 3)
(7, 3)
(0, 4)
(1, 4)
(2, 4)
(3, 4)
(0, 5)
(1, 5)
(2, 5)
(3, 5)
(4, 6)
(5, 6)
(6, 6)
(7, 6)
(4, 7)
(5, 7)
(6, 7)
(7, 7)
(0, 8)
(1, 8)
(2, 8)
(3, 8)
(0, 9)
(1, 9)
(2, 9)
(3, 9)
(4, 10)
(5, 10)
(6, 10)
(7, 10)
(4, 11)
(5, 11)
(6, 11)
(7, 11)
(8, 0)
(9, 0)
(10, 0)
(11, 0)
(8, 1)
(9, 1)
(10, 1)
(11, 1)
(0, 2)
(1, 2)
(2, 2)
(3, 2)
(0, 3)
(1, 3)
(2, 3)
(3, 3)
(8, 4)
(9, 4)
(10, 4)
(11, 4)
(8, 5)
(9, 5)
(10, 5)
(11, 5)
(0, 6)
(1, 6)
(2, 6)
(3, 6)
(0, 7)
(1, 7)
(2, 7)
(3, 7)
(8, 8)
(9, 8)
(10, 8)
(11, 8)
(8, 9)
(9, 9)
(10, 9)
(11, 9)
(4, 0)
(5, 0)
(6, 0)
(7, 0)
(4, 1)
(5, 1)
(6, 1)
(7, 1)
(0, 10)
(1, 10)
(2, 10)
(3, 10)
(0, 11)
(1, 11)
(2, 11)
(3, 11)
(8, 2)
(9, 2)
(10, 2)
(11, 2)
(8, 3)
(9, 3)
(10, 3)
(11, 3)
(4, 4)
(5, 4)
(6, 4)
(7, 4)
(4, 5)
(5, 5)
(6, 5)
(7, 5)
(8, 6)
(9, 6)
(10, 6)
(11, 6)
(8, 7)
(9, 7)
(10, 7)
(11, 7)
(4, 8)
(5, 8)
(6, 8)
(7, 8)
(4, 9)
(5, 9)
(6, 9)
(7, 9)
(8, 10)
(9, 10)
(10, 10)
(11, 10)
(8, 11)
(9, 11)
(10, 11)
(11, 11)
(0, 0, 0.000000)
(1, 0, 0.000000)
(2, 0, 0.000000)
(3, 0, 0.000000)
(0, 1, 0.000000)
(1, 1, 0.000000)
(2, 1, 0.000000)
(3, 1, 0.000000)
(8, 0, 0.000000)
(9, 0, 0.000000)
(10, 0, 0.000000)
(11, 0, 0.000000)
(8, 1, 0.000000)
(9, 1, 0.000000)
(10, 1, 0.000000)
(11, 1, 0.000000)
(0, 10, 0.000000)
(1, 10, 0.000000)
(2, 10, 0.000000)
(3, 10, 0.000000)
(0, 11, 0.000000)
(1, 11, 0.000000)
(2, 11, 0.000000)
(3, 11, 0.000000)
(8, 10, 0.000000)
(9, 10, 0.000000)
(10, 10, 0.000000)
(11, 10, 0.000000)
(8, 11, 0.000000)
(9, 11, 0.000000)
(10, 11, 0.000000)
(11, 11, 0.000000)
(0, 2, 0.000000)
(1, 2, 0.000000)
(2, 2, 0.000000)
(3, 2, 0.000000)
(0, 3, 0.000000)
(1, 3, 0.000000)
(2, 3, 0.000000)
(3, 3, 0.000000)
(4, 0, 0.000000)
(5, 0, 0.000000)
(6, 0, 0.000000)
(7, 0, 0.000000)
(4, 1, 0.000000)
(5, 1, 0.000000)
(6, 1, 0.000000)
(7, 1, 0.000000)
(8, 2, 0.000000)
(9, 2, 0.000000)
(10, 2, 0.000000)
(11, 2, 0.000000)
(8, 3, 0.000000)
(9, 3, 0.000000)
(10, 3, 0.000000)
(11, 3, 0.000000)
(0, 8, 0.000000)
(1, 8, 0.000000)
(2, 8, 0.000000)
(3, 8, 0.000000)
(0, 9, 0.000000)
(1, 9, 0.000000)
(2, 9, 0.000000)
(3, 9, 0.000000)
(4, 10, 0.000000)
(5, 10, 0.000000)
(6, 10, 0.000000)
(7, 10, 0.000000)
(4, 11, 0.000000)
(5, 11, 0.000000)
(6, 11, 0.000000)
(7, 11, 0.000000)
(0, 4, 0.000000)
(1, 4, 0.000000)
(2, 4, 0.000000)
(3, 4, 0.000000)
(0, 5, 0.000000)
(1, 5, 0.000000)
(2, 5, 0.000000)
(3, 5, 0.000000)
(8, 8, 0.000000)
(9, 8, 0.000000)
(10, 8, 0.000000)
(11, 8, 0.000000)
(8, 9, 0.000000)
(9, 9, 0.000000)
(10, 9, 0.000000)
(11, 9, 0.000000)
(4, 2, 0.000000)
(5, 2, 0.000000)
(6, 2, 0.000000)
(7, 2, 0.000000)
(4, 3, 0.000000)
(5, 3, 0.000000)
(6, 3, 0.000000)
(7, 3, 0.000000)
(0, 6, 0.000000)
(1, 6, 0.000000)
(2, 6, 0.000000)
(3, 6, 0.000000)
(0, 7, 0.000000)
(1, 7, 0.000000)
(2, 7, 0.000000)
(3, 7, 0.000000)
(8, 4, 0.000000)
(9, 4, 0.000000)
(10, 4, 0.000000)
(11, 4, 0.000000)
(8, 5, 0.000000)
(9, 5, 0.000000)
(10, 5, 0.000000)
(11, 5, 0.000000)
(8, 6, 0.000000)
(9, 6, 0.000000)
(10, 6, 0.000000)
(11, 6, 0.000000)
(8, 7, 0.000000)
(9, 7, 0.000000)
(10, 7, 0.000000)
(11, 7, 0.000000)
(4, 8, 0.000000)
(5, 8, 0.000000)
(6, 8, 0.000000)
(7, 8, 0.000000)
(4, 9, 0.000000)
(5, 9, 0.000000)
(6, 9, 0.000000)
(7, 9, 0.000000)
(4, 4, 0.000000)
(5, 4, 0.000000)
(6, 4, 0.000000)
(7, 4, 0.000000)
(4, 5, 0.000000)
(5, 5, 0.000000)
(6, 5, 0.000000)
(7, 5, 0.000000)
(4, 6, 0.000000)
(5, 6, 0.000000)
(6, 6, 0.000000)
(7, 6, 0.000000)
(4, 7, 0.000000)
(5, 7, 0.000000)
(6, 7, 0.000000)
(7, 7, 0.000000)
```
## Actual printf output
```
Number of available platforms: 5
Platform: Intel(R) OpenCL HD Graphics
   Required device was found.
(4, 2)
(5, 2)
(6, 2)
(7, 2)
(4, 3)
(5, 3)
(6, 3)
(7, 3)
(0, 0)
(1, 0)
(2, 0)
(3, 0)
(0, 1)
(1, 1)
(2, 1)
(3, 1)
(4, 6)
(5, 6)
(6, 6)
(7, 6)
(4, 7)
(5, 7)
(6, 7)
(7, 7)
(0, 4)
(1, 4)
(2, 4)
(3, 4)
(0, 5)
(1, 5)
(2, 5)
(3, 5)
(4, 10)
(5, 10)
(6, 10)
(7, 10)
(4, 11)
(5, 11)
(6, 11)
(7, 11)
(0, 8)
(1, 8)
(2, 8)
(3, 8)
(0, 9)
(1, 9)
(2, 9)
(3, 9)
(8, 0)
(9, 0)
(10, 0)
(11, 0)
(8, 1)
(9, 1)
(10, 1)
(11, 1)
(0, 2)
(1, 2)
(2, 2)
(3, 2)
(0, 3)
(1, 3)
(2, 3)
(3, 3)
(8, 4)
(9, 4)
(10, 4)
(11, 4)
(8, 5)
(9, 5)
(10, 5)
(11, 5)
(0, 6)
(1, 6)
(2, 6)
(3, 6)
(0, 7)
(1, 7)
(2, 7)
(3, 7)
(8, 8)
(9, 8)
(10, 8)
(11, 8)
(8, 9)
(9, 9)
(10, 9)
(11, 9)
(4, 0)
(5, 0)
(6, 0)
(7, 0)
(4, 1)
(5, 1)
(6, 1)
(7, 1)
(0, 10)
(1, 10)
(2, 10)
(3, 10)
(0, 11)
(1, 11)
(2, 11)
(3, 11)
(8, 2)
(9, 2)
(10, 2)
(11, 2)
(8, 3)
(9, 3)
(10, 3)
(11, 3)
(4, 4)
(5, 4)
(6, 4)
(7, 4)
(4, 5)
(5, 5)
(6, 5)
(7, 5)
(8, 6)
(9, 6)
(10, 6)
(11, 6)
(8, 7)
(9, 7)
(10, 7)
(11, 7)
(4, 8)
(5, 8)
(6, 8)
(7, 8)
(4, 9)
(5, 9)
(6, 9)
(7, 9)
(8, 10)
(9, 10)
(10, 10)
(11, 10)
(8, 11)
(9, 11)
(10, 11)
(11, 11)
```

