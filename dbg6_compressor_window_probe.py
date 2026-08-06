import sys, torch, torch_npu
sys.path.insert(0, "/home/x50061890/vllm-ascend")
import test_compressor_split as T
import itertools

torch.manual_seed(0)
t = T.build_case(1, 4, 4096, 0)
# one-hot x: kv row i = wkv[:, i]
t["x"].zero_()
for i in range(4):
    t["x"][i, i] = 1.0
# deterministic weights: wkv[j, i] = (j*4+i)*0.01 ; wgate=0, ape=0
j = torch.arange(1024, device="npu").unsqueeze(1).float()
i4 = torch.arange(7168, device="npu").unsqueeze(0).float()
t["wkv_w"] = torch.randn(1024, 7168, device="npu").bfloat16() * 0.05
t["wgate_w"].zero_()
t["ape"].zero_()
state0 = t["state_cache"].clone()
t["state_cache"] = state0.clone(); cmp_ref = T.run_fused(t)
t["state_cache"] = state0.clone(); cmp_new = T.run_split(t)

kv = t["wkv_w"].float()[:, :4].t().view(4, 2, 512)  # row i = token i
w = t["norm_w"].float()
def unnorm(out):  # invert rms up to positive scalar
    v = out / w[:out.shape[0]]
    return v / v.norm()
ref, new = unnorm(cmp_ref[0].float()[:448]), unnorm(cmp_new[0].float()[:448])
# candidate means over row subsets of coff0/coff1
best = []
for coff in (0, 1):
    for r in range(1, 5):
        for rows in itertools.combinations(range(4), r):
            m = kv[list(rows), coff, :][: , :448].mean(dim=0)
            m = m / m.norm()
            best.append(((ref - m).abs().max().item(), coff, rows, "ref"))
            best.append(((new - m).abs().max().item(), coff, rows, "new"))
best.sort()
for b in best[:6]:
    print(f"diff={b[0]:.4f} coff={b[1]} rows={b[2]} vs={b[3]}")

best_new = sorted([b for b in best if b[3] == "new"])[:4]
print("BEST FOR NEW:")
for b in best_new:
    print(f"diff={b[0]:.4f} coff={b[1]} rows={b[2]} vs={b[3]}")
# also: does new match ref up to scalar? (direction)
print("new vs ref direction:", (new - ref).abs().max().item())

# permutation check: is split's nope a rolled/permuted version of fused's?
r = cmp_ref[0].float()[:448]; n = cmp_new[0].float()[:448]
for k in range(-7, 8):
    rolled = torch.roll(r, k*64, dims=0)
    d = (rolled - n).abs().max().item()
    if d < 0.5: print(f"ROLL {k*64}: maxdiff {d:.4f}")
# block-permutation: split chunk i equals fused chunk j?
rc = r.view(7, 64); nc = n.view(7, 64)
perm = []
for i in range(7):
    ds = [(rc[j] - nc[i]).abs().max().item() for j in range(7)]
    perm.append((i, int(torch.argmin(torch.tensor(ds)).item()), min(ds)))
print("chunk mapping (split_i -> fused_j, diff):", perm)
