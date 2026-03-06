# deep-sparse-index

brief overview of deep sparse attention, a sparse variant of deepseeks multi latent attention, also used in glm-5 (happy belated chinese new year !) taken from flashinfer-bench definitions (https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)

**dsa sparse index:**

- Inputs: `q_index_fp8 [batch_size, 64, 128]` (fp8), `k_index_cache_fp8 [num_pages, 64, 1, 132]` (fp8 + 1 fp32 scale), `weights [batch_size, 64]` (f32), `seq_len`, `block_table [B , max_pages]` (int32)
- Output: `topk_indices [batch_size, 2048]` int32

**dsa sparse attention:**

- Inputs: `q_nope [T, 16, 512]`, `q_pe [T, 16, 64]`, paged ckv_cache / kpe_cache, `sparse_indices [T, 2048]`, `sm_scale`
- Outputs: `output [T, 16, 512]` bf16, `lse [T, 16]` f32

---

dsa is split into 2 phases, this post focuses on the index phase. this is the phase where we actually select the tokens that get staged for computation from the current sequence length, and poist this we have the fixed dim main attention phase where we compute attention scores for all the topk index selected tokens. 
for the flash infer problem definition topk is selected as 2048 but i think it can be arbitrary. in real terms this would be all 2048 tokens in the chat history that are scored as being relevant to the next token generation. looking at the inputs for the index stage more closely, we have a query index of head size 64 and dim 128 that gets loaded once per decode, and a key cache of dim 132 (actual dim of they key value per token is 128 , the extra 4 bytes are a single fp32 scale we multiple each key value by). note the key cache is not stored by batch but has batches sharing a combined page table (block_table) that maps token batches to physical mem addresses, storing (batch_size (64), key_dim (128) in blocks, customary paged attention design. seq_len is self explanatory. from an inference perspective the main win of this design is that per token memory loads , which dominate in long contexts, are cut down to 132 fp8 vals. Regarding the actual design, my rough understanding is that q_index and k_index work something like a budget long context attention score computation,
we store less encoded information (128 dims vs 512 = less information stored per each token), have more heads (64 vs 16 = more judging criterion for relevant tokens). also no positional encoding in this stage. relative to the k index, all the other loads are fairly trivial in size. from a compute perspective, we just loop over the ceiling of all 64 token length batches, multiply all per token key values by its scale factor, then the dominant calculation is our query * key matmul  `(head_dim (64), key_dim (128)) * (key_dim , seq_len)` reducing the key_dim across head_dim and seq_len. after this, we `relu(head_dim, seq_len)` (remove negative values), and then weight each heads importance `(head_dim , seq_len) * (weights, 1)` , and finally reduce the head dims ( `(head_dim, seq_len).sum(dim(0))` ), and then select final 2048 most relevant tokens (`topk((seq_len,1))`).

regarding occupancy, we mostly want to partition an equal amount of key tiles among ctas, and fully saturate the gpus sms. for this we can just compute a static schedule in host,

thinking from a Blackwell / B200 architecture perspective, the shapes align nicely with `tcgen05.mma.kind::f8f6f4` (using fp8). per the ptx isa (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape ), a batch of `(head_dim, key_dim)  (key_dim , batch_size)` , maps cleanly onto an mma with m , n = 64 and k = 32 , looping over K 4 times. per SM , a B200 has 512 cols of tmem, so for an mma that writes 64 tmem columns of output each stage, we can buffer up to 8 num stages. as we only need to load k each stage, this fits with tons of smem to spare. in code :

```cuda
if (warp_id == kMmaWarp && elect_sync()) {
    for (int local = 0; local < num_my_tiles; ++local) {

        wait_mbars(tma_mbar , tmem_reuse_mbar);

        if (valid_tokens > 0) {
            for (int ki = 0; ki < kMmaIters; ++ki) {
                tcgen05_mma_f8f6f4(tmem_d, q_desc, k_desc, kIdesc, (ki > 0));
            }
            tcgen05_commit(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        } else {
            mbarrier_arrive(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}
```

(this is simplifed pseudo-ish code with some stuff removed / combined for brevity)

here we loop over all asigned tiles, wait for each tma stage to complete , and for tiles >= num stages we also wait on the epilogue to be done reading tmem slots, and

for data loading, query indices are small and all shares, we can load this once at the start

```cuda
if (warp_id == kProducerWarp && elect_sync()) {
    constexpr int q_bytes = kNumHeads * kHeadDim;
    mbarrier_arrive_expect_tx(addr.q_mbar, q_bytes);
    tma_3d_gmem2smem(addr.q_stage, &q_fp8_tmap, 0, 0, b, addr.q_mbar, 0ULL);
    mbarrier_wait_parity(addr.q_mbar, 0);
}
```

(note: we dont need to use tensor maps for this, it might be quicker just to direct load, i mostly TMA'd so that swizzle patterns match later k_ vectors , means we can reuse smem descriptors for mma)

and the main key vectors we loop over , double buffered with the main MMA warp.

```cuda
// ---------------- Producer warp ----------------
if (warp_id == kProducerWarp && elect_sync()) {
    for (int local = 0; local < num_my_tiles; ++local) {

        mbar_wait(add.epi_mbar)
        prepare_stage_metadata_and_scale_tail()

        if (valid_tokens > 0) {

            mbarrier_arrive_expect_tx(mbar, payload_bytes + scale_bytes);
            tma_3d_gmem2smem(payload_dst, &k_fp8_tmap, 0, 0, page_idx, mbar, 0ULL);
            tma_2d_gmem2smem(scale_dst, &k_scale_tmap, 0, page_idx, mbar, 0ULL);
        } else {
            mbarrier_arrive(addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}
```

per stage we have separate tensor maps and loads for both the main tokens and the scales

the last major component of the first index phase is epilogue, which is loading results from tmem, appling a fused scale * relu * weight multiply to all the 64 x 64 (head , token) results of the mma. from a speed standpoint, this phase is by far the biggest bottleneck. per hardware rules (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-issue-granularity), each thread in a cta is limited to accessing only one lane of tmem memory, so we need a full waprgoup ( 4 ctas of cta % 0 - 3) for each ep stage, although we can just pack as many warpgroups as we can for each stage and assign sequential mma stages to a different warpgroup to hide latency. given the hard cap of 32 warps / cta , this gives us up to 7 epilogue warp groups / mma . this is a somewhat crude strat but does give notable speedups, although epi is still the bottleneck. regarding the actual logic , this is more involved than prior stages. as a starting point , on mma completion each  64 x 64 (head , token) is stored in tmem in a 128 x 64 layout, heads corresponding to lanes and tokens corresponding to columns. our final desired score is 64 token values, with the fused op applied and all head dims summed. note, only the first 16 threads of each warp actually getting useful head dim data in the ld op (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f). this maps directly onto `tcgen05.ld.ld.sync.aligned.16x64b.x32` instruction well (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld) (64 bit depth = 2 values so x32)  although this instruction caused issues with crashes / register pressure, instead we can just loop over mulitple chunks using smaller x16 or x8 variants. Testing either of these didnt show much difference, although smaller variants were worse.

directly loading all 64 vals to a single thread crashes wtih reg overflow or somethign, although we can loop over with `.ld x32 x16` variants, testing between the 2 didnt show much difference , smaller variants performed worse (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld).

```cuda
#pragma unroll
for (int ci = 0; ci < kNumChunks; ++ci) {
    const int tok_off = ci * kChunk;
    float vals[kChunk];
    tcgen05_ld_16x64b_8(lane_base, tmem_col_base + tok_off, vals);
    tcgen05_wait_ld();

    if (active_lane) {
        // all compute and shuffle logic here
    }
}
```

loading and applying the fused multiplication to these values :

```cuda
if (active_lane) {  // active lan  <= 16
    #pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        const int tok = tok_off + t;
        float v = 0.0f;
        if (tok < valid_tokens) {
            v = fmaxf(vals[t] * stage_scale[tok], 0.0f) * w;
        }
        vals[t] = v;
    }
```

now each warp stores 16 heads x 16 tokens , we can compute the final token value first by warp shuffling the 16 heads across all 4 warp groups, and then storing the intermediate final sum in a (64 , 4) smem buffer

```cuda
#pragma unroll
for (int off = 8; off > 0; off >>= 1) {
    #pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        vals[t] += __shfl_down_sync(0x0000FFFF, vals[t], off, 16);
    }
}
```

```cuda
if (lane16 == 0) {  // only single thread holds vals after warp shuffle
    #pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        const int tok = tok_off + t;
        if (tok < valid_tokens) {
            const int part_idx =
                ((stage * kNumEpSlots + ep_slot) * kStageTokens) + tok;
            s.stage_tile_raw[part_idx] = vals[t];
        }
    }
}
```

and the final result is us storing this result to global, and signalling mbar completion so we can reuse tmem cols / smem staging area. There's probably some notable savings to be made in epilogue, I made some attempts at using tcgen05. 16x64 , which avoids redundant loads, and would allow us to cut down num stages for the same reg pressure, but had awkward warp shuffle semantics and doesn't seem to yield major speedups. There's also maybe an opportunity to fold some of the loops, altogether we have 3 loops inside the main token chunk loop, as well as some other optimizations.

---

lastly, we need to find the 2048 highest scoring values from those we computed. while tesing i just computed `torch.topk()` over the results, which worked for correctness but was somewhat slow. final result is to just vibe code a new cublas radix histogram kernel, and graph capture it with the prior kernel, which at lest gave 50% ~ is speedup over using torch eager. I'm not completely sure how this works, but the basic idea is launch a cta for every batch we receive, and progressively stream in all the computed values for this batch into smem buffers, find the smallest value from the current buffer + existing topk vals by decomposing all our existing vals into buckets based on the values leading bits until we find the smallest value from the current list that meets our topk threshold (2048 in our case) , then another phase , again per smem tile, where we sort and keep all values >= this , then another after we have soft-sorted all tiles that hard sorts the values in descending order.

Again, definite speedups here, first by just fusing the kernel with the main index kernel , we could keep per batch sort by having the first (batch_len) ctas of the index kernel, after their final epilogue, wait on a global atomic that all ctas write to on completion, and then computing radix sort here. this would at least allow us to l2 evict last hint all the output values / indices we write in the prior phase. for large seq_len, with each tile handling well over 2048 tokens, its also possible we could overlap a pre topk in the index kernel with the other pipelined tma / mma / epi stages and write the indices of only the best 2048 tokens we compute to a buffer that our main batch cta reads, which if it didn't stall the rest of the pipeline would get rid of a lot of topk work for the final sort. i did some testing on this, early results is it seems hard for it to keep speed unless we pump number of threads, which also slows epilogue.

computing against the pytorch ref, we get a few swapped indice values in the topk , because of the way the torch ref upscales the values to fp32 and multiples the head weight over all 128 key dim prior to the main mma. this is mathematically equivalent, but has some small diffs because of fp accumulation or something, anyway the actual topk values we produce are the same , we just have a few positional differences, so i decided to just ignore it for now. final speedup ranges from 2.5 - 15 x over the pytorch reference and about 2-3 x over a torch version i made that doesn't pointlessly split the final loop into batches. considering we have fp8 vs fp32 data movement and fp8 Blackwell tensor cores, this is rather depressingly small for now, and points to what i said about epilogue / topk being the main bottleneck. all code / benchmarking on both flash infer and custom bench can be found here https://github.com/jIab-b/deep-sparse-attention  

---

