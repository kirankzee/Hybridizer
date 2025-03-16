using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;

namespace HybridizerExample
{
    class Program
    {
        // This attribute marks the method as an entry point for CUDA execution
        [EntryPoint]
        public static void VectorAdd(float[] a, float[] b, float[] c, int n)
        {
            // Get the thread index
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            
            // Make sure we don't go out of bounds
            if (i < n)
            {
                c[i] = a[i] + b[i];
            }
        }

        // CPU implementation of vector addition
        public static void VectorAddCPU(float[] a, float[] b, float[] c, int n)
        {
            for (int i = 0; i < n; i++)
            {
                c[i] = a[i] + b[i];
            }
        }

        // CPU implementation using parallel processing
        public static void VectorAddCPUParallel(float[] a, float[] b, float[] c, int n)
        {
            Parallel.For(0, n, i =>
            {
                c[i] = a[i] + b[i];
            });
        }

        // Benchmark class to store performance results
        public class BenchmarkResult
        {
            public int VectorSize { get; set; }
            public double CpuTimeMs { get; set; }
            public double CpuParallelTimeMs { get; set; }
            public double GpuTimeMs { get; set; }
            public double CpuToGpuSpeedup { get; set; }
            public double CpuParallelToGpuSpeedup { get; set; }
            public double GpuMemoryTransferTimeMs { get; set; }
            public double GpuKernelTimeMs { get; set; }
        }

        // Run a single benchmark for a specific vector size
        private static BenchmarkResult RunBenchmark(int vectorSize, int iterations, int warmupIterations)
        {
            // Create input vectors and output vectors
            float[] a = new float[vectorSize];
            float[] b = new float[vectorSize];
            float[] c_cpu = new float[vectorSize];
            float[] c_cpu_parallel = new float[vectorSize];
            float[] c_gpu = new float[vectorSize];
            
            // Initialize vectors with random data
            Random rand = new Random(42);
            for (int i = 0; i < vectorSize; i++)
            {
                a[i] = (float)rand.NextDouble();
                b[i] = (float)rand.NextDouble();
            }

            // Warm up CPU (regular)
            for (int i = 0; i < warmupIterations; i++)
            {
                VectorAddCPU(a, b, c_cpu, vectorSize);
            }
            
            // Benchmark CPU (regular)
            Stopwatch cpuTimer = new Stopwatch();
            cpuTimer.Start();
            for (int i = 0; i < iterations; i++)
            {
                VectorAddCPU(a, b, c_cpu, vectorSize);
            }
            cpuTimer.Stop();
            double cpuTimeMs = cpuTimer.ElapsedMilliseconds / (double)iterations;
            
            // Warm up CPU (parallel)
            for (int i = 0; i < warmupIterations; i++)
            {
                VectorAddCPUParallel(a, b, c_cpu_parallel, vectorSize);
            }
            
            // Benchmark CPU (parallel)
            Stopwatch cpuParallelTimer = new Stopwatch();
            cpuParallelTimer.Start();
            for (int i = 0; i < iterations; i++)
            {
                VectorAddCPUParallel(a, b, c_cpu_parallel, vectorSize);
            }
            cpuParallelTimer.Stop();
            double cpuParallelTimeMs = cpuParallelTimer.ElapsedMilliseconds / (double)iterations;
            
            // Configure CUDA execution
            int threadsPerBlock = 256;
            int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
            
            // Create Hybridizer runner
            HybRunner runner = HybRunner.Cuda("VectorAddition_CUDA.dll").SetDistrib(blocksPerGrid, 1, threadsPerBlock, 1, 1, 0);
            
            // Wrap the current instance
            dynamic wrapper = runner.Wrap(new Program());
            
            // Warm up GPU
            for (int i = 0; i < warmupIterations; i++)
            {
                wrapper.VectorAdd(a, b, c_gpu, vectorSize);
            }
            
            // Benchmark GPU (including memory transfer)
            Stopwatch gpuTimer = new Stopwatch();
            gpuTimer.Start();
            for (int i = 0; i < iterations; i++)
            {
                wrapper.VectorAdd(a, b, c_gpu, vectorSize);
            }
            gpuTimer.Stop();
            double gpuTimeMs = gpuTimer.ElapsedMilliseconds / (double)iterations;
            
            // Estimate memory transfer time (this is an approximation as Hybridizer abstracts the memory transfers)
            double dataSize = vectorSize * 3 * sizeof(float); // 3 arrays of floats
            double memoryBandwidthGBps = 12.0; // Approximate for a mid-range GPU, adjust as needed
            double estimatedTransferTimeMs = (dataSize / (memoryBandwidthGBps * 1e9)) * 1000 * 2; // *2 for bidirectional transfer
            
            // Estimate kernel execution time
            double kernelTimeMs = gpuTimeMs - estimatedTransferTimeMs;
            if (kernelTimeMs < 0) kernelTimeMs = 0.01; // Prevent negative values
            
            // Validate results
            bool correct = true;
            for (int i = 0; i < vectorSize; i++)
            {
                if (Math.Abs(c_gpu[i] - c_cpu[i]) > 1e-5)
                {
                    correct = false;
                    Console.WriteLine($"Error at position {i}: GPU={c_gpu[i]}, CPU={c_cpu[i]}");
                    break;
                }
            }
            
            if (!correct)
            {
                Console.WriteLine($"Validation failed for vector size {vectorSize}");
            }
            
            return new BenchmarkResult
            {
                VectorSize = vectorSize,
                CpuTimeMs = cpuTimeMs,
                CpuParallelTimeMs = cpuParallelTimeMs,
                GpuTimeMs = gpuTimeMs,
                CpuToGpuSpeedup = cpuTimeMs / gpuTimeMs,
                CpuParallelToGpuSpeedup = cpuParallelTimeMs / gpuTimeMs,
                GpuMemoryTransferTimeMs = estimatedTransferTimeMs,
                GpuKernelTimeMs = kernelTimeMs
            };
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Vector Addition Benchmark using Hybridizer C# CUDA");
            Console.WriteLine("=================================================");
            Console.WriteLine();
            
            try
            {
                // Define vector sizes to benchmark
                int[] vectorSizes = { 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000 };
                
                // Number of iterations for each benchmark
                int iterations = 10;
                int warmupIterations = 3;
                
                List<BenchmarkResult> results = new List<BenchmarkResult>();
                
                Console.WriteLine($"Running benchmarks with {iterations} iterations per size (plus {warmupIterations} warmup iterations)");
                Console.WriteLine();
                
                // Run benchmarks for each vector size
                foreach (int size in vectorSizes)
                {
                    Console.WriteLine($"Benchmarking vector size: {size:N0}");
                    BenchmarkResult result = RunBenchmark(size, iterations, warmupIterations);
                    results.Add(result);
                    
                    // Print individual result
                    Console.WriteLine($"  CPU Time: {result.CpuTimeMs:F2} ms");
                    Console.WriteLine($"  CPU Parallel Time: {result.CpuParallelTimeMs:F2} ms");
                    Console.WriteLine($"  GPU Time (total): {result.GpuTimeMs:F2} ms");
                    Console.WriteLine($"  GPU Memory Transfer (est.): {result.GpuMemoryTransferTimeMs:F2} ms");
                    Console.WriteLine($"  GPU Kernel (est.): {result.GpuKernelTimeMs:F2} ms");
                    Console.WriteLine($"  Speedup (CPU → GPU): {result.CpuToGpuSpeedup:F2}x");
                    Console.WriteLine($"  Speedup (CPU Parallel → GPU): {result.CpuParallelToGpuSpeedup:F2}x");
                    Console.WriteLine();
                }
                
                // Print summary table
                Console.WriteLine("Benchmark Summary");
                Console.WriteLine("=================");
                Console.WriteLine("Vector Size | CPU (ms) | CPU Parallel (ms) | GPU (ms) | CPU→GPU | CPUParallel→GPU");
                Console.WriteLine("------------------------------------------------------------------------");
                
                foreach (var result in results)
                {
                    Console.WriteLine($"{result.VectorSize,10:N0} | {result.CpuTimeMs,8:F2} | {result.CpuParallelTimeMs,16:F2} | {result.GpuTimeMs,7:F2} | {result.CpuToGpuSpeedup,7:F2}x | {result.CpuParallelToGpuSpeedup,15:F2}x");
                }
                
                // Generate performance analysis
                Console.WriteLine();
                Console.WriteLine("Performance Analysis");
                Console.WriteLine("===================");
                
                // Calculate averages
                double avgCpuToGpuSpeedup = results.Average(r => r.CpuToGpuSpeedup);
                double avgCpuParallelToGpuSpeedup = results.Average(r => r.CpuParallelToGpuSpeedup);
                
                Console.WriteLine($"Average CPU → GPU Speedup: {avgCpuToGpuSpeedup:F2}x");
                Console.WriteLine($"Average CPU Parallel → GPU Speedup: {avgCpuParallelToGpuSpeedup:F2}x");
                
                // Find best case
                var bestCase = results.OrderByDescending(r => r.CpuToGpuSpeedup).First();
                Console.WriteLine($"Best Speedup: {bestCase.CpuToGpuSpeedup:F2}x with vector size {bestCase.VectorSize:N0}");
                
                // Analyze scaling
                Console.WriteLine();
                Console.WriteLine("Scaling Analysis:");
                Console.WriteLine("As vector size increases:");
                
                if (results.Last().CpuToGpuSpeedup > results.First().CpuToGpuSpeedup * 1.5)
                {
                    Console.WriteLine("- GPU performance advantage increases with larger data sizes");
                }
                else if (results.Last().CpuToGpuSpeedup < results.First().CpuToGpuSpeedup * 0.8)
                {
                    Console.WriteLine("- GPU performance advantage decreases with larger data sizes");
                }
                else
                {
                    Console.WriteLine("- GPU performance advantage remains relatively stable across data sizes");
                }
                
                // Memory transfer impact
                double avgMemoryTransferPercentage = results.Average(r => r.GpuMemoryTransferTimeMs / r.GpuTimeMs * 100);
                Console.WriteLine($"- Memory transfer overhead: ~{avgMemoryTransferPercentage:F1}% of total GPU time");
                
                if (avgMemoryTransferPercentage > 50)
                {
                    Console.WriteLine("  (Memory transfer is the bottleneck for this operation)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            
            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
