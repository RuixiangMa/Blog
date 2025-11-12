#!/usr/bin/env python3
"""
PyTorch缓存产物保存和加载演示,通过torch.compiler.save_cache_artifacts()和torch.compiler.load_cache_artifacts()
来保存和加载编译期间的缓存产物，包括：
- 编译的Triton内核代码
- 自动调优的配置参数
- 图优化结果
- 缓存键和元数据信息
"""

import torch
import time
import tempfile
import os
import pickle
from typing import Optional, Tuple


def my_function(x):
    """测试函数：矩阵运算和激活函数组合"""
    return x.sin() @ x.cos() + x.tanh()


def complex_function(x, y):
    """更复杂的测试函数：包含多个操作"""
    z = torch.matmul(x, y.T)
    z = torch.nn.functional.relu(z)
    z = torch.nn.functional.gelu(z)
    return torch.sum(z, dim=-1, keepdim=True)


def benchmark_function(func, input_data, num_runs=10, warmup_runs=3):
    """基准测试函数"""
    # warmup
    for _ in range(warmup_runs):
        _ = func(input_data)
    
    # 同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 正式测试
    start_time = time.time()
    for _ in range(num_runs):
        result = func(input_data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time, result


def save_cache_to_file(artifacts: Tuple[bytes, any], filename: str):
    """将缓存产物保存到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"缓存产物已保存到: {filename}")


def load_cache_from_file(filename: str) -> Optional[Tuple[bytes, any]]:
    """从文件加载缓存产物"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"缓存文件不存在: {filename}")
        return None


def analyze_cache_info(cache_info):
    """分析缓存信息"""
    if cache_info is None:
        print("没有缓存信息")
        return
    
    print("\n=== 缓存信息分析 ===")
    print(f"缓存产物类型: {type(cache_info)}")
    
    # 检查不同类型的缓存产物
    artifact_types = [
        'inductor_artifacts',
        'autotune_artifacts', 
        'aot_autograd_artifacts',
        'pgo_artifacts',
        'precompile_artifacts'
    ]
    
    total_artifacts = 0
    for artifact_type in artifact_types:
        if hasattr(cache_info, artifact_type):
            artifacts = getattr(cache_info, artifact_type)
            if artifacts:
                print(f"{artifact_type}: {len(artifacts)} 个产物")
                for i, artifact_key in enumerate(artifacts[:3]):  # 只显示前3个
                    print(f"  - {artifact_key}")
                if len(artifacts) > 3:
                    print(f"  ... 还有 {len(artifacts) - 3} 个")
                total_artifacts += len(artifacts)
    
    print(f"缓存产物总数: {total_artifacts}")


def main():
    print("=== PyTorch 缓存产物保存和加载演示 ===\n")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    input_tensor = torch.randn(100, 100, device=device)
    input_tensor2 = torch.randn(100, 100, device=device)
    
    # 第一次编译 - 没有缓存
    print("\n2. 第一次编译（无缓存）...")
    torch._dynamo.reset()  # 重置Dynamo缓存
    
    start_compile = time.time()
    compiled_fn = torch.compile(my_function, backend="inductor")
    first_result = compiled_fn(input_tensor)
    first_compile_time = time.time() - start_compile
    
    print(f"第一次编译时间: {first_compile_time:.4f}秒")
    
    # 运行基准测试
    print("\n3. 基准测试（第一次编译版本）...")
    first_avg_time, _ = benchmark_function(compiled_fn, input_tensor)
    print(f"平均执行时间: {first_avg_time:.6f}秒")
    
    # 保存缓存产物
    print("\n4. 保存缓存产物...")
    artifacts = torch.compiler.save_cache_artifacts()
    
    if artifacts:
        artifact_bytes, cache_info = artifacts
        print(f"缓存产物大小: {len(artifact_bytes)} 字节")
        analyze_cache_info(cache_info)
        
        # 保存到文件
        cache_filename = "/tmp/pytorch_cache_artifacts.pkl"
        save_cache_to_file(artifacts, cache_filename)
    else:
        print("没有可用的缓存产物")
        return
    
    # 重置环境进行第二次测试
    print("\n5. 重置环境进行第二次编译测试...")
    torch._dynamo.reset()
    
    # 加载缓存产物
    print("\n6. 加载缓存产物...")
    loaded_artifacts = load_cache_from_file(cache_filename)
    if loaded_artifacts:
        loaded_cache_info = torch.compiler.load_cache_artifacts(loaded_artifacts[0])
        print("缓存产物加载成功")
        analyze_cache_info(loaded_cache_info)
    
    # 第二次编译 - 有缓存
    print("\n7. 第二次编译（有缓存）...")
    start_compile = time.time()
    compiled_fn2 = torch.compile(my_function, backend="inductor")
    second_result = compiled_fn2(input_tensor)
    second_compile_time = time.time() - start_compile
    
    print(f"第二次编译时间: {second_compile_time:.4f}秒")
    print(f"编译时间减少: {((first_compile_time - second_compile_time) / first_compile_time * 100):.1f}%")
    
    # 验证结果一致性
    print("\n8. 验证结果一致性...")
    if torch.allclose(first_result, second_result, rtol=1e-5, atol=1e-5):
        print("✓ 两次编译结果一致")
    else:
        print("✗ 两次编译结果不一致")
        print(f"结果差异: {torch.max(torch.abs(first_result - second_result))}")
    
    # 测试复杂函数
    print("\n9. 测试复杂函数...")
    torch._dynamo.reset()
    
    # 复杂函数第一次编译
    start_compile = time.time()
    compiled_complex = torch.compile(complex_function, backend="inductor")
    complex_result1 = compiled_complex(input_tensor, input_tensor2)
    first_complex_time = time.time() - start_compile
    
    # 保存复杂函数缓存
    complex_artifacts = torch.compiler.save_cache_artifacts()
    
    # 重置并重新编译
    torch._dynamo.reset()
    if complex_artifacts:
        torch.compiler.load_cache_artifacts(complex_artifacts[0])
    
    start_compile = time.time()
    compiled_complex2 = torch.compile(complex_function, backend="inductor")
    complex_result2 = compiled_complex2(input_tensor, input_tensor2)
    second_complex_time = time.time() - start_compile
    
    print(f"复杂函数第一次编译: {first_complex_time:.4f}秒")
    print(f"复杂函数第二次编译: {second_complex_time:.4f}秒")
    print(f"复杂函数编译加速: {((first_complex_time - second_complex_time) / first_complex_time * 100):.1f}%")
    
    if torch.allclose(complex_result1, complex_result2, rtol=1e-5, atol=1e-5):
        print("✓ 复杂函数结果一致")
    else:
        print("✗ 复杂函数结果不一致")
    
    # 性能对比
    print("\n10. 性能对比分析...")
    print(f"简单函数编译加速: {((first_compile_time - second_compile_time) / first_compile_time * 100):.1f}%")
    print(f"复杂函数编译加速: {((first_complex_time - second_complex_time) / first_complex_time * 100):.1f}%")
    
    # 清理
    if os.path.exists(cache_filename):
        os.remove(cache_filename)
        print(f"\n清理缓存文件: {cache_filename}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()