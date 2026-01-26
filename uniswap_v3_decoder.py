#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, decimal
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests
from eth_abi import decode as abi_decode
from eth_utils import keccak, to_checksum_address

decimal.getcontext().prec = 80

# --- constants ---
WETH9 = to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
ROUTERS = {
    to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564"),
    to_checksum_address("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"),
    to_checksum_address("0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B"),
}

def sel(sig: str) -> bytes:
    return keccak(text=sig)[:4]

# router selectors we decode
SELECTOR_MAP: Dict[bytes, Tuple[str, List[str]]] = {
    sel("exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))"):
        ("exactInputSingle",  ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"]),
    sel("exactOutputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))"):
        ("exactOutputSingle", ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"]),
    sel("exactInput((bytes,address,uint256,uint256,uint256))"):
        ("exactInput",        ["(bytes,address,uint256,uint256,uint256)"]),
    sel("exactOutput((bytes,address,uint256,uint256,uint256))"):
        ("exactOutput",       ["(bytes,address,uint256,uint256,uint256)"]),
    sel("multicall(bytes[])"):          ("multicall", ["bytes[]"]),
    sel("multicall(uint256,bytes[])"):  ("multicall", ["uint256","bytes[]"]),
    sel("sweepToken(address,uint256,address)"):
        ("sweepToken", ["address","uint256","address"]),
    sel("unwrapWETH9(uint256,address)"):
        ("unwrapWETH9", ["uint256","address"]),
    sel("refundETH()"):
        ("refundETH", []),
    # keep UR execute decoding (record only) - no logic change
    sel("execute(bytes,bytes[])"):
        ("urExecute", ["bytes","bytes[]"]),
    sel("execute(bytes,bytes[],uint256)"):
        ("urExecute", ["bytes","bytes[]","uint256"]),
}

SWAP_TOPIC0_HEX = keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()
SWAP_TOPIC0_NOX_LOWER = SWAP_TOPIC0_HEX.lower()  # compare against s0x(topic0).lower()

# --- RPC ---
class RPC:
    """
    Speed optimizations (logic unchanged):
      - reuse a single requests.Session() (keep-alive)
      - memoize eth_call results (safe for token0/token1/decimals calls)
    """
    def __init__(self, url: str, timeout=25):
        self.url, self.timeout, self._id = url, timeout, 1
        self._sess = requests.Session()
        self._eth_call_cache: Dict[Tuple[str, str, str], str] = {}

    def call(self, method: str, params):
        payload = {"jsonrpc":"2.0", "id": self._id, "method": method, "params": params}
        self._id += 1
        r = self._sess.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(f"RPC error: {j['error']}")
        return j["result"]

    def get_tx(self, h): return self.call("eth_getTransactionByHash", [h])
    def get_receipt(self, h): return self.call("eth_getTransactionReceipt", [h])

    def eth_call(self, to, data, block="latest"):
        # cache eth_call for identical (to,data,block) to avoid repeated RPC roundtrips
        key = (to_checksum_address(to), data, block)
        hit = self._eth_call_cache.get(key)
        if hit is not None:
            return hit
        res = self.call("eth_call", [{"to": key[0], "data": data}, block])
        self._eth_call_cache[key] = res
        return res

# --- small utils ---
def s0x(h: str) -> str: return h[2:] if h.startswith("0x") else h
def h2i(h: str) -> int: return int(h, 16)
def chk(a: Optional[str]) -> Optional[str]: return None if a is None else to_checksum_address(a)

def to_hr(x: int, dec: int) -> str:
    d = Decimal(x) / (Decimal(10) ** Decimal(dec))
    s = format(d, "f")
    return (s.rstrip("0").rstrip(".")) if "." in s else s

def decode_input(input_hex: str):
    if not input_hex or input_hex == "0x" or len(input_hex) < 10:
        return None, None
    b = bytes.fromhex(s0x(input_hex))
    info = SELECTOR_MAP.get(b[:4])
    if not info:
        return None, None
    name, types = info
    if not types:
        return name, []
    return name, list(abi_decode(types, b[4:]))

def abi_call_selector(sig: str) -> str:
    return "0x" + sel(sig).hex()

def call_addr(rpc: RPC, contract: str, sig: str) -> str:
    ret = rpc.eth_call(contract, abi_call_selector(sig))
    b = bytes.fromhex(s0x(ret))
    if len(b) < 32: raise RuntimeError("bad eth_call return")
    return to_checksum_address("0x" + b[-20:].hex())

def call_u8(rpc: RPC, contract: str, sig: str) -> int:
    ret = rpc.eth_call(contract, abi_call_selector(sig))
    b = bytes.fromhex(s0x(ret))
    if len(b) < 32: raise RuntimeError("bad eth_call return")
    return int.from_bytes(b[-32:], "big")

def parse_path(p: bytes) -> List[str]:
    if not p or len(p) < 20: return []
    i, toks = 20, [to_checksum_address("0x" + p[:20].hex())]
    while i < len(p):
        if i + 3 > len(p): break
        i += 3
        if i + 20 > len(p): break
        toks.append(to_checksum_address("0x" + p[i:i+20].hex()))
        i += 20
    return toks

@dataclass
class Intent:
    idx: int
    callType: str
    tokenIn: Optional[str]
    tokenOut: Optional[str]
    recipient: Optional[str]
    pathTokens: Optional[List[str]]

@dataclass
class Hop:
    logIndex: int
    pool: str
    tokenIn: str
    tokenOut: str
    amountInInt: int
    amountOutInt: int

def decode_intent(name: str, args: List[Any], idx: int) -> Intent:
    tin = tout = rec = None
    pt = None
    if name in ("exactInputSingle", "exactOutputSingle"):
        p = args[0]
        tin, tout, rec = to_checksum_address(p[0]), to_checksum_address(p[1]), to_checksum_address(p[3])
        return Intent(idx, name, tin, tout, rec, None)

    if name in ("exactInput", "exactOutput"):
        p = args[0]
        pt = parse_path(p[0])
        rec = to_checksum_address(p[1])
        if pt:
            if name == "exactInput":
                tin, tout = pt[0], pt[-1]
            else:
                tout, tin = pt[0], pt[-1]
        return Intent(idx, name, tin, tout, rec, pt)

    return Intent(idx, name, None, None, None, None)

def walk_calls(input_hex: str, depth=0, max_depth=6) -> List[Dict[str, Any]]:
    if depth > max_depth: return []
    name, args = decode_input(input_hex)
    if name is None: return []
    out = [{"name": name, "args": args, "raw": input_hex}]
    if name == "multicall":
        datas = args[0] if len(args) == 1 else (args[1] if len(args) == 2 else None)
        if datas:
            for b in datas:
                out.extend(walk_calls("0x" + b.hex(), depth+1, max_depth))
    return out

def infer_recipient(calls: List[Dict[str, Any]], swap_recipient: Optional[str], token_out: Optional[str]) -> Optional[str]:
    if not swap_recipient: return None
    swap_recipient = to_checksum_address(swap_recipient)
    if swap_recipient not in ROUTERS:
        return swap_recipient

    token_out = to_checksum_address(token_out) if token_out else None
    final = swap_recipient
    for c in calls:
        if c["name"] == "sweepToken":
            token, _, rec = c["args"]
            token, rec = to_checksum_address(token), to_checksum_address(rec)
            if token_out is None or token == token_out:
                final = rec
        if c["name"] == "unwrapWETH9":
            _, rec = c["args"]
            final = to_checksum_address(rec)
    return final

def has_unwrap(calls: List[Dict[str, Any]]) -> bool:
    return any(c["name"] == "unwrapWETH9" for c in calls)

def extract_hops(rpc: RPC, logs: List[Dict[str, Any]]) -> List[Hop]:
    """
    Speed optimizations (logic unchanged):
      - cache pool->(token0,token1) so each pool queried once (instead of per Swap log)
    """
    out: List[Hop] = []
    pool_tok_cache: Dict[str, Tuple[str, str]] = {}

    for lg in logs:
        topics = lg.get("topics", [])
        if not topics:
            continue
        if s0x(topics[0]).lower() != SWAP_TOPIC0_NOX_LOWER:
            continue

        pool = to_checksum_address(lg["address"])
        logi = h2i(lg.get("logIndex", "0x0"))

        data = bytes.fromhex(s0x(lg["data"]))
        a0, a1, _, _, _ = abi_decode(["int256","int256","uint160","uint128","int24"], data)
        a0, a1 = int(a0), int(a1)

        tt = pool_tok_cache.get(pool)
        if tt is None:
            t0 = call_addr(rpc, pool, "token0()")
            t1 = call_addr(rpc, pool, "token1()")
            tt = (t0, t1)
            pool_tok_cache[pool] = tt
        t0, t1 = tt

        if a0 > 0 and a1 < 0:
            tin, tout, ain, aout = t0, t1, a0, -a1
        elif a1 > 0 and a0 < 0:
            tin, tout, ain, aout = t1, t0, a1, -a0
        else:
            continue

        out.append(Hop(logi, pool, tin, tout, ain, aout))

    out.sort(key=lambda x: x.logIndex)
    return out

def build_candidates(hops: List[Hop], max_len=8) -> List[List[Hop]]:
    seqs: List[List[Hop]] = []
    n = len(hops)
    for i in range(n):
        chain = [hops[i]]
        j = i + 1
        while j < n and len(chain) < max_len and chain[-1].tokenOut == hops[j].tokenIn:
            chain.append(hops[j]); j += 1
        for L in range(1, len(chain)+1):
            seqs.append(chain[:L])
    return seqs

def seq_tokens(seq: List[Hop]) -> List[str]:
    if not seq: return []
    toks = [seq[0].tokenIn]
    for h in seq: toks.append(h.tokenOut)
    return toks

def score(seq: List[Hop], intent: Optional[Intent]) -> Tuple[int, int]:
    if not seq: return (-10_000, 0)
    amt_in = seq[0].amountInInt
    st = seq_tokens(seq)
    in_tok, out_tok = st[0], st[-1]

    if intent is None or (intent.tokenIn is None and intent.tokenOut is None and not intent.pathTokens):
        return (0, amt_in)

    sc = 0
    if intent.tokenIn and in_tok == intent.tokenIn: sc += 10
    if intent.tokenOut and out_tok == intent.tokenOut: sc += 10

    if intent.pathTokens and len(intent.pathTokens) >= 2:
        pt = intent.pathTokens
        if st == pt: sc += 100
        elif st == list(reversed(pt)): sc += 80
        if len(seq) == len(pt) - 1: sc += 15
        else: sc -= 5

    return (sc, amt_in)

def decode_uniswap_v3_swap(rpc: RPC, tx_hash: str, return_all=False):
    tx = rpc.get_tx(tx_hash)
    if not tx: raise RuntimeError("Transaction not found")
    receipt = rpc.get_receipt(tx_hash)
    if not receipt: raise RuntimeError("Receipt not found (pending?)")

    sender = chk(tx.get("from"))
    input_hex = tx.get("input", "0x")
    tx_value_wei = h2i(tx.get("value", "0x0"))

    calls = walk_calls(input_hex)

    intents: List[Intent] = []
    for idx, c in enumerate(calls):
        if c["name"] in ("exactInputSingle","exactOutputSingle","exactInput","exactOutput"):
            intents.append(decode_intent(c["name"], c["args"], idx))

    hops = extract_hops(rpc, receipt.get("logs", []))
    if not hops:
        raise RuntimeError("No Uniswap V3 Pool Swap events found in receipt logs.")

    candidates = build_candidates(hops, max_len=8)

    best, best_sc, best_intent = None, (-10_000, -1), None
    if intents:
        for it in intents:
            for seq in candidates:
                sc = score(seq, it)
                if sc > best_sc:
                    best_sc, best, best_intent = sc, seq, it
    else:
        for seq in candidates:
            sc = score(seq, None)
            if sc > best_sc:
                best_sc, best, best_intent = sc, seq, None

    if not best:
        raise RuntimeError("Could not select a swap sequence from Swap events.")

    token_in, token_out = best[0].tokenIn, best[-1].tokenOut
    amt_in_int, amt_out_int = best[0].amountInInt, best[-1].amountOutInt

    swap_rec = best_intent.recipient if best_intent else None
    recipient = infer_recipient(calls, swap_rec, token_out) or swap_rec or sender

    # keep ETH flags (commented output like your current)
    native_in = (tx_value_wei > 0 and token_in == WETH9)
    native_out = (token_out == WETH9 and has_unwrap(calls))

    # Speed: cache token decimals across the whole decode (logic unchanged)
    dec_cache: Dict[str, int] = {}

    def decimals_of_cached(tok: str) -> int:
        tok = to_checksum_address(tok)
        v = dec_cache.get(tok)
        if v is not None:
            return v
        try:
            v = call_u8(rpc, tok, "decimals()")
        except Exception:
            v = 18
        dec_cache[tok] = v
        return v

    din, dout = decimals_of_cached(token_in), decimals_of_cached(token_out)
    result = {
        "sender": sender,
        "recipient": recipient,
        "tokenIn": token_in,
        "tokenOut": token_out,
        "amountIn": to_hr(amt_in_int, din),
        "amountOut": to_hr(amt_out_int, dout),
        # "nativeIn": bool(native_in),
        # "nativeOut": bool(native_out),
    }

    if not return_all:
        return result

    all_swaps = []
    for seq in candidates:
        tin, tout = seq[0].tokenIn, seq[-1].tokenOut
        all_swaps.append({
            "tokenIn": tin,
            "tokenOut": tout,
            "amountIn": to_hr(seq[0].amountInInt, decimals_of_cached(tin)),
            "amountOut": to_hr(seq[-1].amountOutInt, decimals_of_cached(tout)),
            "hopCount": len(seq),
            "pathTokens": seq_tokens(seq),
            "firstLogIndex": seq[0].logIndex,
            "lastLogIndex": seq[-1].logIndex,
        })

    return {
        **result,
        "_allSwapCandidates": all_swaps,
        "_selected": {
            "score": best_sc[0],
            "tieBreakerAmountInInt": best_sc[1],
            "intentUsed": None if not best_intent else {
                "callType": best_intent.callType,
                "tokenIn": best_intent.tokenIn,
                "tokenOut": best_intent.tokenOut,
                "recipient": best_intent.recipient,
                "pathTokens": best_intent.pathTokens,
            }
        }
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python uniswap_v3_decoder.py <tx_hash> [--all]", file=sys.stderr)
        sys.exit(1)

    tx_hash = sys.argv[1].strip()
    return_all = ("--all" in sys.argv[2:])

    if not (tx_hash.startswith("0x") and len(tx_hash) == 66):
        print("Invalid tx hash", file=sys.stderr)
        sys.exit(1)

    rpc_url = os.environ.get("RPC_URL", "").strip()
    if not rpc_url:
        print("Please set RPC_URL environment variable (Ethereum mainnet JSON-RPC endpoint).", file=sys.stderr)
        sys.exit(1)

    rpc = RPC(rpc_url)
    try:
        out = decode_uniswap_v3_swap(rpc, tx_hash, return_all=return_all)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
