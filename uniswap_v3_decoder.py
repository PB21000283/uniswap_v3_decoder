#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import json
import decimal
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests
from eth_abi import decode as abi_decode
from eth_utils import keccak, to_checksum_address

decimal.getcontext().prec = 80

# =============================================================================
# Constants
# =============================================================================

WETH9 = to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")

# Known router addresses (used for recipient inference only; decoding uses selectors)
ROUTERS = {
    to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564"),  # Uniswap V3 SwapRouter
    to_checksum_address("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"),  # SwapRouter02
    to_checksum_address("0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B"),  # Universal Router
    to_checksum_address("0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD"),  # Universal Router v2
}

def sel(sig: str) -> bytes:
    """4-byte function selector."""
    return keccak(text=sig)[:4]

# Uniswap V3 Pool Swap topic0
SWAP_TOPIC0_NOX_LOWER = keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
).hex().lower()

# Universal Router command codes (lower 5 bits)
UR_V3_SWAP_EXACT_IN = 0x00
UR_V3_SWAP_EXACT_OUT = 0x01

# =============================================================================
# Selector decode registry (IMPORTANT: support multiple type candidates per selector)
# =============================================================================

# We store selector -> list of (name, [types...]) candidates
# Decode tries candidates in order until abi_decode succeeds.
SELECTOR_CANDIDATES: Dict[bytes, List[Tuple[str, List[str]]]] = {}

def add_selector(sig: str, name: str, types: List[str]) -> None:
    SELECTOR_CANDIDATES.setdefault(sel(sig), []).append((name, types))

# SwapRouter (deadline in struct)
add_selector(
    "exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))",
    "exactInputSingle",
    ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
)
add_selector(
    "exactOutputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))",
    "exactOutputSingle",
    ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
)
add_selector(
    "exactInput((bytes,address,uint256,uint256,uint256))",
    "exactInput",
    ["(bytes,address,uint256,uint256,uint256)"],
)
add_selector(
    "exactOutput((bytes,address,uint256,uint256,uint256))",
    "exactOutput",
    ["(bytes,address,uint256,uint256,uint256)"],
)

# SwapRouter02 (NO deadline in struct)
add_selector(
    "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))",
    "exactInputSingle",
    ["(address,address,uint24,address,uint256,uint256,uint160)"],
)
add_selector(
    "exactOutputSingle((address,address,uint24,address,uint256,uint256,uint160))",
    "exactOutputSingle",
    ["(address,address,uint24,address,uint256,uint256,uint160)"],
)
add_selector(
    "exactInput((bytes,address,uint256,uint256))",
    "exactInput",
    ["(bytes,address,uint256,uint256)"],
)
add_selector(
    "exactOutput((bytes,address,uint256,uint256))",
    "exactOutput",
    ["(bytes,address,uint256,uint256)"],
)

# multicall
add_selector("multicall(bytes[])", "multicall", ["bytes[]"])
add_selector("multicall(uint256,bytes[])", "multicall", ["uint256", "bytes[]"])

# payout helpers (Router / Router02)
add_selector("sweepToken(address,uint256,address)", "sweepToken", ["address", "uint256", "address"])
add_selector("unwrapWETH9(uint256,address)", "unwrapWETH9", ["uint256", "address"])
add_selector("refundETH()", "refundETH", [])

# Universal Router execute (we decode commands ourselves)
add_selector("execute(bytes,bytes[])", "urExecute", ["bytes", "bytes[]"])
add_selector("execute(bytes,bytes[],uint256)", "urExecute", ["bytes", "bytes[]", "uint256"])

# =============================================================================
# RPC client (optimized)
# =============================================================================

class RPC:
    """
    Optimizations:
      - reuse requests.Session() keep-alive
      - memoize eth_call results (safe for token0/token1/decimals calls)
    """
    def __init__(self, url: str, timeout: int = 25):
        self.url = url
        self.timeout = timeout
        self._id = 1
        self._sess = requests.Session()
        self._eth_call_cache: Dict[Tuple[str, str, str], str] = {}

    def call(self, method: str, params: list) -> Any:
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self._id += 1
        r = self._sess.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(f"RPC error: {j['error']}")
        return j["result"]

    def get_tx(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        return self.call("eth_getTransactionByHash", [tx_hash])

    def get_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        return self.call("eth_getTransactionReceipt", [tx_hash])

    def eth_call(self, to: str, data: str, block: str = "latest") -> str:
        key = (to_checksum_address(to), data, block)
        hit = self._eth_call_cache.get(key)
        if hit is not None:
            return hit
        res = self.call("eth_call", [{"to": key[0], "data": data}, block])
        self._eth_call_cache[key] = res
        return res

# =============================================================================
# Utilities
# =============================================================================

def s0x(h: str) -> str:
    return h[2:] if isinstance(h, str) and h.startswith("0x") else h

def h2i(h: str) -> int:
    return int(h, 16)

def chk(a: Optional[str]) -> Optional[str]:
    return None if a is None else to_checksum_address(a)

def to_hr(x: int, dec: int) -> str:
    d = Decimal(x) / (Decimal(10) ** Decimal(dec))
    s = format(d, "f")
    return (s.rstrip("0").rstrip(".")) if "." in s else s

def abi_call_selector(sig: str) -> str:
    return "0x" + sel(sig).hex()

def call_addr(rpc: RPC, contract: str, sig: str) -> str:
    ret = rpc.eth_call(contract, abi_call_selector(sig))
    b = bytes.fromhex(s0x(ret))
    if len(b) < 32:
        raise RuntimeError("bad eth_call return")
    return to_checksum_address("0x" + b[-20:].hex())

def call_u8(rpc: RPC, contract: str, sig: str) -> int:
    ret = rpc.eth_call(contract, abi_call_selector(sig))
    b = bytes.fromhex(s0x(ret))
    if len(b) < 32:
        raise RuntimeError("bad eth_call return")
    return int.from_bytes(b[-32:], "big")

def parse_path(p: bytes) -> List[str]:
    """
    Uniswap V3 path encoding:
      token(20) + fee(3) + token(20) + fee(3) + token(20) ...
    """
    if not p or len(p) < 20:
        return []
    i = 20
    toks = [to_checksum_address("0x" + p[:20].hex())]
    while i < len(p):
        if i + 3 > len(p):
            break
        i += 3
        if i + 20 > len(p):
            break
        toks.append(to_checksum_address("0x" + p[i:i+20].hex()))
        i += 20
    return toks

def decode_input(input_hex: str) -> Tuple[Optional[str], Optional[list]]:
    """
    Decode input data using selector candidates; try all type candidates until one works.
    Returns (name, args) or (None, None) if unknown/un-decodable.
    """
    if not input_hex or input_hex == "0x" or len(input_hex) < 10:
        return None, None

    b = bytes.fromhex(s0x(input_hex))
    selector = b[:4]
    cands = SELECTOR_CANDIDATES.get(selector)
    if not cands:
        return None, None

    for name, types in cands:
        if not types:
            return name, []
        try:
            return name, list(abi_decode(types, b[4:]))
        except Exception:
            continue

    return None, None

# =============================================================================
# Data structures
# =============================================================================

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

# =============================================================================
# Calldata intent decoding
# =============================================================================

def decode_intent(name: str, args: List[Any], idx: int) -> Intent:
    tin = tout = rec = None
    pt = None

    if name in ("exactInputSingle", "exactOutputSingle"):
        p = args[0]
        # both router versions share leading fields: tokenIn, tokenOut, fee, recipient, ...
        tin = to_checksum_address(p[0])
        tout = to_checksum_address(p[1])
        rec = to_checksum_address(p[3])
        return Intent(idx, name, tin, tout, rec, None)

    if name in ("exactInput", "exactOutput"):
        p = args[0]
        pt = parse_path(p[0])
        rec = to_checksum_address(p[1])
        if pt:
            if name == "exactInput":
                tin, tout = pt[0], pt[-1]
            else:
                # exactOutput path is reversed in meaning
                tout, tin = pt[0], pt[-1]
        return Intent(idx, name, tin, tout, rec, pt)

    return Intent(idx, name, None, None, None, None)

def decode_universal_router_intents(commands: bytes, inputs: List[bytes], base_idx: int) -> List[Intent]:
    """
    Decode Universal Router execute() commands into Intent list.
    We only care about V3 swap commands (cmd_type 0x00, 0x01).
    """
    out: List[Intent] = []
    for i, cmd in enumerate(commands):
        if i >= len(inputs):
            break
        cmd_type = cmd & 0x1f  # lower 5 bits
        if cmd_type == UR_V3_SWAP_EXACT_IN:
            # (address recipient, uint256 amountIn, uint256 amountOutMin, bytes path, bool payerIsUser)
            try:
                recipient, _, _, path, _payer = abi_decode(
                    ["address", "uint256", "uint256", "bytes", "bool"],
                    inputs[i]
                )
                pt = parse_path(path)
                if pt:
                    out.append(Intent(
                        base_idx + i,
                        "urV3SwapExactIn",
                        pt[0],
                        pt[-1],
                        to_checksum_address(recipient),
                        pt
                    ))
            except Exception:
                pass

        elif cmd_type == UR_V3_SWAP_EXACT_OUT:
            # (address recipient, uint256 amountOut, uint256 amountInMax, bytes path, bool payerIsUser)
            try:
                recipient, _, _, path, _payer = abi_decode(
                    ["address", "uint256", "uint256", "bytes", "bool"],
                    inputs[i]
                )
                pt = parse_path(path)
                if pt:
                    # exactOut path is reversed in meaning
                    out.append(Intent(
                        base_idx + i,
                        "urV3SwapExactOut",
                        pt[-1],
                        pt[0],
                        to_checksum_address(recipient),
                        pt
                    ))
            except Exception:
                pass
    return out

def walk_calls(input_hex: str, depth: int = 0, max_depth: int = 6) -> List[Dict[str, Any]]:
    """
    Recursively walk nested calls (multicall).
    Returns a flat list of decoded call dicts: {"name","args","raw"}.
    """
    if depth > max_depth:
        return []
    name, args = decode_input(input_hex)
    if name is None:
        return []
    out = [{"name": name, "args": args, "raw": input_hex}]
    if name == "multicall":
        datas = args[0] if len(args) == 1 else (args[1] if len(args) == 2 else None)
        if datas:
            for b in datas:
                out.extend(walk_calls("0x" + b.hex(), depth + 1, max_depth))
    return out

# =============================================================================
# Recipient inference
# =============================================================================

def infer_recipient(calls: List[Dict[str, Any]], swap_recipient: Optional[str], token_out: Optional[str]) -> Optional[str]:
    """
    For Router/Router02, recipient may be the router itself, then later sweepToken/unwrapWETH9
    sends outputs to the final user. We look for these.
    If swap_recipient is not a known router, we consider it final already.
    """
    if not swap_recipient:
        return None
    swap_recipient = to_checksum_address(swap_recipient)

    if swap_recipient not in ROUTERS:
        return swap_recipient

    token_out = to_checksum_address(token_out) if token_out else None
    final = swap_recipient

    for c in calls:
        if c["name"] == "sweepToken":
            token, _, rec = c["args"]
            token = to_checksum_address(token)
            rec = to_checksum_address(rec)
            if token_out is None or token == token_out:
                final = rec
        if c["name"] == "unwrapWETH9":
            _, rec = c["args"]
            final = to_checksum_address(rec)

    return final

def has_unwrap(calls: List[Dict[str, Any]]) -> bool:
    return any(c["name"] == "unwrapWETH9" for c in calls)

# =============================================================================
# Log parsing: extract V3 swap hops from pool Swap events
# =============================================================================

def extract_hops(rpc: RPC, logs: List[Dict[str, Any]]) -> List[Hop]:
    """
    Extract V3 Swap events from receipt logs.
    We decode amounts and determine direction using token0/token1 from pool contract.
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
        try:
            a0, a1, _, _, _ = abi_decode(["int256", "int256", "uint160", "uint128", "int24"], data)
        except Exception:
            continue
        a0, a1 = int(a0), int(a1)

        tt = pool_tok_cache.get(pool)
        if tt is None:
            t0 = call_addr(rpc, pool, "token0()")
            t1 = call_addr(rpc, pool, "token1()")
            tt = (t0, t1)
            pool_tok_cache[pool] = tt
        t0, t1 = tt

        # Pool perspective:
        #   positive amount = token paid into pool (input)
        #   negative amount = token sent out of pool (output)
        if a0 > 0 and a1 < 0:
            tin, tout, ain, aout = t0, t1, a0, -a1
        elif a1 > 0 and a0 < 0:
            tin, tout, ain, aout = t1, t0, a1, -a0
        else:
            continue

        out.append(Hop(logi, pool, tin, tout, ain, aout))

    out.sort(key=lambda x: x.logIndex)
    return out

# =============================================================================
# Swap sequence selection
# =============================================================================

def build_candidates(hops: List[Hop], max_len: int = 8) -> List[List[Hop]]:
    """
    Build candidate sequences by chaining consecutive hops if tokenOut matches next tokenIn.
    Adds all subchains as candidates.
    """
    seqs: List[List[Hop]] = []
    n = len(hops)
    for i in range(n):
        chain = [hops[i]]
        j = i + 1
        while j < n and len(chain) < max_len and chain[-1].tokenOut == hops[j].tokenIn:
            chain.append(hops[j])
            j += 1
        for L in range(1, len(chain) + 1):
            seqs.append(chain[:L])
    return seqs

def seq_tokens(seq: List[Hop]) -> List[str]:
    if not seq:
        return []
    toks = [seq[0].tokenIn]
    for h in seq:
        toks.append(h.tokenOut)
    return toks

def score(seq: List[Hop], intent: Optional[Intent]) -> Tuple[int, int]:
    """
    Score a sequence against an intent.
    Returns (score, tie_break_amountIn).
    """
    if not seq:
        return (-10_000, 0)

    amt_in = seq[0].amountInInt
    tokens = seq_tokens(seq)
    in_tok, out_tok = tokens[0], tokens[-1]

    if intent is None or (intent.tokenIn is None and intent.tokenOut is None and not intent.pathTokens):
        return (0, amt_in)

    sc = 0
    if intent.tokenIn and in_tok == intent.tokenIn:
        sc += 10
    if intent.tokenOut and out_tok == intent.tokenOut:
        sc += 10

    if intent.pathTokens and len(intent.pathTokens) >= 2:
        pt = intent.pathTokens
        if tokens == pt:
            sc += 100
        elif tokens == list(reversed(pt)):
            sc += 80
        if len(seq) == len(pt) - 1:
            sc += 15
        else:
            sc -= 5

    return (sc, amt_in)

# =============================================================================
# Main decode
# =============================================================================

def decode_uniswap_v3_swap(rpc: RPC, tx_hash: str, return_all: bool = False) -> Dict[str, Any]:
    tx = rpc.get_tx(tx_hash)
    if not tx:
        raise RuntimeError("Transaction not found")
    receipt = rpc.get_receipt(tx_hash)
    if not receipt:
        raise RuntimeError("Receipt not found (pending?)")

    if receipt.get("status", "0x1") == "0x0":
        raise RuntimeError("Transaction failed (reverted)")

    sender = chk(tx.get("from"))
    input_hex = tx.get("input", "0x")
    tx_value_wei = h2i(tx.get("value", "0x0"))

    # Decode calls (multicall nesting)
    calls = walk_calls(input_hex)

    # Decode intents
    intents: List[Intent] = []
    for idx, c in enumerate(calls):
        if c["name"] in ("exactInputSingle", "exactOutputSingle", "exactInput", "exactOutput"):
            intents.append(decode_intent(c["name"], c["args"], idx))
        elif c["name"] == "urExecute":
            # execute(bytes commands, bytes[] inputs [,uint256 deadline])
            args = c["args"]
            if len(args) >= 2:
                commands = args[0]
                inputs = args[1]
                # use idx*10_000 so UR internal order doesn't collide with other call indices
                intents.extend(decode_universal_router_intents(commands, inputs, base_idx=idx * 10_000))

    # Extract swap hops (source of truth)
    hops = extract_hops(rpc, receipt.get("logs", []))
    if not hops:
        raise RuntimeError("No Uniswap V3 Pool Swap events found in receipt logs.")

    candidates = build_candidates(hops, max_len=8)

    # Pick best candidate
    best: Optional[List[Hop]] = None
    best_sc: Tuple[int, int] = (-10_000, -1)
    best_intent: Optional[Intent] = None

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

    token_in = best[0].tokenIn
    token_out = best[-1].tokenOut
    amt_in_int = best[0].amountInInt
    amt_out_int = best[-1].amountOutInt

    # Recipient inference
    swap_rec = best_intent.recipient if best_intent else None
    recipient = infer_recipient(calls, swap_rec, token_out) or swap_rec or sender

    # Native ETH hints (optional; not included in output per prompt)
    _native_in = (tx_value_wei > 0 and token_in == WETH9)
    _native_out = (token_out == WETH9 and has_unwrap(calls))

    # Decimals cache
    dec_cache: Dict[str, int] = {}

    def decimals_of(tok: str) -> int:
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

    din = decimals_of(token_in)
    dout = decimals_of(token_out)

    result = {
        "sender": sender,
        "recipient": recipient,
        "tokenIn": token_in,
        "tokenOut": token_out,
        "amountIn": to_hr(amt_in_int, din),
        "amountOut": to_hr(amt_out_int, dout),
    }

    if not return_all:
        return result

    # Debug: all candidates
    all_swaps = []
    for seq in candidates:
        tin, tout = seq[0].tokenIn, seq[-1].tokenOut
        all_swaps.append({
            "tokenIn": tin,
            "tokenOut": tout,
            "amountIn": to_hr(seq[0].amountInInt, decimals_of(tin)),
            "amountOut": to_hr(seq[-1].amountOutInt, decimals_of(tout)),
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

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python uniswap_v3_decoder.py <tx_hash> [--all]", file=sys.stderr)
        print('Example: export RPC_URL="https://ethereum-rpc.publicnode.com"', file=sys.stderr)
        sys.exit(1)

    tx_hash = sys.argv[1].strip()
    return_all = ("--all" in sys.argv[2:])

    if not (tx_hash.startswith("0x") and len(tx_hash) == 66):
        print("Invalid tx hash (expect 0x + 64 hex chars).", file=sys.stderr)
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
