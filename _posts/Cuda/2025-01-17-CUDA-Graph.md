---
categories: [Cuda]
tags: CUDA Graph pytorch SGLang
---
将一系列 CUDA 内核被定义和封装为一个单元，即一个算子图，而不是一系列单独启动的算子。它提供了一种通过单个 CPU 操作 launch 多个 GPU 算子的机制，从而减少 launch 开销。
{% highlight c++ %} 
// 不使用cuda graph
for(int istep=0; istep<NSTEP; istep++){
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
  }
  cudaStreamSynchronize(stream);
}

// 使用cuda graph
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;
for(int istep=0; istep<NSTEP; istep++){
   //这部分的构建部分时间较长，但是只有一次，后续直接触发cudaGraphLaunch
  if(!graphCreated){
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated=true;
  }
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
}
{% endhighlight %}
适用范围：kernel，memcpy等

pytorch中使用方法如下：
{% highlight python %} 
g = torch.cuda.CUDAGraph()
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()
{% endhighlight %}

在SGLang中，调用init_cuda_graphs执行，目前的版本仅generation model支持

{% highlight python %} 
def init_cuda_graphs(self):
    """Capture cuda graphs."""
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    self.cuda_graph_runner = None

    if not self.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return

    if self.server_args.disable_cuda_graph:
        return

    tic = time.time()
    logger.info("Capture cuda graph begin. This can take up to several minutes.")
    // 生成cuda图
    self.cuda_graph_runner = CudaGraphRunner(self)
    logger.info(f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s")
{% endhighlight %}