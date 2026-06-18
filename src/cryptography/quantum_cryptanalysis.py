"""
Quantum Cryptanalysis Module
=============================
# ID: CRYPTO-001
# Purpose: Demonstrate quantum algorithms that threaten classical cryptographic
#          schemes, and quantum-safe alternatives. Covers Grover's attack on
#          symmetric encryption (AES key search) and the Shor's algorithm
#          framework for RSA public-key factoring.
#
# Requirement: Simulate and benchmark quantum cryptanalytic circuits against
#              classical security parameters.
#
# References:
#   - Grover (1996), arXiv:quant-ph/9605043
#   - Shor  (1994), arXiv:quant-ph/9508027
#   - NIST PQC Standardization (2024), FIPS 203/204/205
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumCryptanalysis:
    """
    # ID: CRYPTO-CLASS-001
    # Purpose: Collection of quantum circuits and classical analyses that
    #          demonstrate quantum threats to classical cryptographic primitives.
    #
    # Inputs:   None at construction - methods are self-contained.
    # Outputs:  Per-method result dicts with circuit objects and analysis data.
    # Side Effects: Prints formatted analysis to stdout.
    # Assumptions: AerSimulator available; no real IBM token required for demos.
    """

    def __init__(self):
        # ID: CRYPTO-INIT-001
        # Purpose: Initialize local quantum simulator.
        self.simulator = AerSimulator()
        logger.info("Initialized QuantumCryptanalysis module")

    # ------------------------------------------------------------------
    # GROVER'S ATTACK ON SYMMETRIC KEYS
    # ------------------------------------------------------------------
    def grover_key_search_demo(self, key_bits: int = 3) -> Dict:
        """
        # ID: CRYPTO-GROVER-001
        # Requirement: Demonstrate Grover's algorithm searching a symmetric key
        #              space of size 2^key_bits, achieving O(sqrt(2^n)) queries.
        # Purpose: Illustrate why AES-128 provides only 64-bit quantum security
        #          and why NIST recommends AES-256 for post-quantum resistance.
        #
        # Inputs:
        #   key_bits (int): Number of key bits to search. Range 1-8 for demo.
        #                   Real AES uses 128/192/256 bits.
        # Outputs:
        #   dict with keys:
        #     'target_key'        (int)  : The secret key being searched.
        #     'grover_iterations' (int)  : Number of Grover iterations applied.
        #     'counts'            (dict) : Measurement histogram.
        #     'circuit'           (QuantumCircuit): The built circuit.
        #     'classical_queries' (int)  : Expected classical queries (worst case).
        #     'quantum_queries'   (int)  : Expected Grover queries.
        #
        # Preconditions: key_bits >= 1.
        # Postconditions: Returns result dict; prints analysis to stdout.
        # Failure Modes: key_bits > 15 will be slow; capped at 8 for demo.
        # Verification: Measure output should peak at target_key bitstring.
        """
        key_bits = min(key_bits, 8)  # Guard: cap at 8 for demo speed
        key_space = 2 ** key_bits

        # Choose a random "secret" key to search for
        target_key = np.random.randint(0, key_space)
        target_bits = format(target_key, f'0{key_bits}b')

        print(f"\n{'='*60}")
        print(f"  GROVER KEY SEARCH DEMO  ({key_bits}-bit key space)")
        print(f"{'='*60}")
        print(f"  Key space size        : {key_space} possible keys")
        print(f"  Secret target key     : {target_key} (binary: {target_bits})")

        classical_queries = key_space // 2          # expected classical average
        quantum_queries = math.ceil(math.pi / 4 * math.sqrt(key_space))
        speedup = classical_queries / max(quantum_queries, 1)

        print(f"  Classical avg queries : {classical_queries}")
        print(f"  Grover queries needed : {quantum_queries}")
        print(f"  Quantum speedup       : {speedup:.1f}x")

        # --- Build Grover circuit ---
        qc = QuantumCircuit(key_bits, key_bits)

        # Step 1: Uniform superposition over all keys
        qc.h(range(key_bits))
        qc.barrier()

        # Step 2: Apply Grover iterations
        for _ in range(quantum_queries):
            # Oracle: phase-flip the target key state
            # Mark target by flipping qubits where target bit = 0, applying
            # multi-controlled Z, then unflipping
            for i, bit in enumerate(reversed(target_bits)):
                if bit == '0':
                    qc.x(i)
            if key_bits == 1:
                qc.z(0)
            else:
                qc.h(key_bits - 1)
                qc.mcx(list(range(key_bits - 1)), key_bits - 1)
                qc.h(key_bits - 1)
            for i, bit in enumerate(reversed(target_bits)):
                if bit == '0':
                    qc.x(i)
            qc.barrier()

            # Diffusion operator: 2|s><s| - I
            qc.h(range(key_bits))
            qc.x(range(key_bits))
            qc.h(key_bits - 1)
            if key_bits > 1:
                qc.mcx(list(range(key_bits - 1)), key_bits - 1)
            qc.h(key_bits - 1)
            qc.x(range(key_bits))
            qc.h(range(key_bits))
            qc.barrier()

        # Measure
        qc.measure(range(key_bits), range(key_bits))

        # Run
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=1024)
        counts = job.result().get_counts()

        # Most probable result should be target_bits
        top_result = max(counts, key=counts.get)
        found_key = int(top_result, 2)
        success = (found_key == target_key)

        print(f"\n  Measurement results   : {dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")
        print(f"  Most probable result  : {top_result} = key {found_key}")
        print(f"  Key found correctly   : {'YES' if success else 'NO'}")

        return {
            'target_key': target_key,
            'grover_iterations': quantum_queries,
            'counts': counts,
            'circuit': qc,
            'classical_queries': classical_queries,
            'quantum_queries': quantum_queries,
        }

    # ------------------------------------------------------------------
    # SHOR'S ALGORITHM FRAMEWORK (Period Finding)
    # ------------------------------------------------------------------
    def shors_period_finding_demo(self, N: int = 15, a: int = 7) -> Dict:
        """
        # ID: CRYPTO-SHOR-001
        # Requirement: Demonstrate the period-finding subroutine of Shor's
        #              algorithm which underlies RSA factoring.
        # Purpose: Show that RSA security collapses with a fault-tolerant
        #          quantum computer. The period r of f(x)=a^x mod N satisfies
        #          gcd(a^(r/2) +/- 1, N) = prime factors of N.
        #
        # Inputs:
        #   N (int): The number to factor. Demo uses N=15 (3x5), the smallest
        #            RSA-like semiprime demonstrable on a quantum computer.
        #   a (int): Coprime base for modular exponentiation. Must gcd(a,N)=1.
        # Outputs:
        #   dict with 'N', 'a', 'period', 'factors', 'classical_complexity',
        #             'quantum_complexity', 'security_implications'.
        #
        # Preconditions: N must be composite; gcd(a, N) must equal 1.
        # Postconditions: Returns factors of N derived from period.
        # Failure Modes: If r is odd or a^(r/2) = -1 mod N, retry with new a.
        # Verification: product of factors should equal N.
        """
        from math import gcd

        print(f"\n{'='*60}")
        print(f"  SHOR'S ALGORITHM FRAMEWORK  (factoring N={N})")
        print(f"{'='*60}")

        # Validate inputs
        if gcd(a, N) != 1:
            raise ValueError(f"a={a} and N={N} must be coprime. gcd={gcd(a, N)}")

        print(f"  Factoring N           : {N}")
        print(f"  Coprime base a        : {a}")
        print(f"  Classical complexity  : O(exp(n^(1/3))) where n=log2(N)={int(math.log2(N)+1)} bits")
        print(f"  Quantum complexity    : O((log N)^3) = O({int(math.log2(N))**3})")

        # --- Classical simulation of period finding (quantum would use QFT) ---
        # In a real quantum implementation, this uses quantum phase estimation
        # with the QFT. Here we simulate classically to show the result,
        # then explain what the quantum circuit would do.
        period = None
        for r in range(1, N):
            if pow(a, r, N) == 1:
                period = r
                break

        print(f"\n  Period r of f(x)=a^x mod N : {period}")
        print(f"  (Quantum computer finds this via QFT-based phase estimation)")

        factors = []
        if period and period % 2 == 0:
            x = pow(a, period // 2, N)
            f1 = gcd(x + 1, N)
            f2 = gcd(x - 1, N)
            for f in [f1, f2]:
                if 1 < f < N:
                    factors.append(f)

        factors = list(set(factors))
        verified = len(factors) >= 2 and (factors[0] * factors[1] == N)

        print(f"  Candidate factors     : {factors}")
        print(f"  Verification (f1*f2=N): {'PASS' if verified else 'PARTIAL'}")

        # Security implications
        rsa_bits = {512: "broken (1999)", 1024: "broken (2010s)", 2048: "vulnerable (future QC)",
                    4096: "safe today, vulnerable at ~4000 logical qubits"}
        print(f"\n  RSA Security Implications:")
        for bits, status in rsa_bits.items():
            print(f"    RSA-{bits:4d} : {status}")
        print(f"\n  Post-quantum alternative : CRYSTALS-Kyber (NIST FIPS 203, 2024)")
        print(f"  Post-quantum alternative : CRYSTALS-Dilithium (NIST FIPS 204, 2024)")

        return {
            'N': N,
            'a': a,
            'period': period,
            'factors': factors,
            'classical_complexity': f'O(exp({int(math.log2(N)+1)}^(1/3)))',
            'quantum_complexity': f'O({int(math.log2(N))**3})',
            'security_implications': rsa_bits,
        }

    # ------------------------------------------------------------------
    # QUANTUM KEY DISTRIBUTION (BB84) SIMULATION
    # ------------------------------------------------------------------
    def bb84_qkd_simulation(self, num_bits: int = 20) -> Dict:
        """
        # ID: CRYPTO-BB84-001
        # Requirement: Simulate the BB84 quantum key distribution protocol,
        #              demonstrating quantum-secure key exchange that is
        #              information-theoretically secure against any eavesdropper.
        # Purpose: Show the constructive side of quantum cryptography - not just
        #          breaking classical crypto, but building provably secure QKD.
        #
        # Inputs:
        #   num_bits (int): Number of raw bits Alice sends. Typical: 20-100.
        # Outputs:
        #   dict with 'alice_key', 'bob_key', 'sifted_key', 'key_rate',
        #             'eavesdrop_detected' fields.
        #
        # Preconditions: num_bits >= 4.
        # Failure Modes: With eavesdropper simulation, error rate ~25% per qubit.
        # Verification: Without eavesdropper, alice_key == bob_key on sifted bits.
        """
        print(f"\n{'='*60}")
        print(f"  BB84 QKD PROTOCOL SIMULATION  ({num_bits} raw bits)")
        print(f"{'='*60}")

        # Alice generates random bits and bases
        alice_bits  = np.random.randint(0, 2, num_bits)
        alice_bases = np.random.randint(0, 2, num_bits)  # 0=Z-basis, 1=X-basis
        bob_bases   = np.random.randint(0, 2, num_bits)

        bob_results = []
        circuits = []

        for i in range(num_bits):
            qc = QuantumCircuit(1, 1)
            # Alice prepares qubit
            if alice_bits[i] == 1:
                qc.x(0)                         # |1> in Z, |-> in X
            if alice_bases[i] == 1:
                qc.h(0)                         # rotate to X basis

            # Bob measures in his chosen basis
            if bob_bases[i] == 1:
                qc.h(0)
            qc.measure(0, 0)

            transpiled = transpile(qc, self.simulator)
            result = self.simulator.run(transpiled, shots=1).result()
            bob_results.append(int(list(result.get_counts().keys())[0]))
            circuits.append(qc)

        # Sift: keep only bits where Alice and Bob used the same basis
        sifted_alice = []
        sifted_bob   = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_results[i])

        sifted_len  = len(sifted_alice)
        key_rate    = sifted_len / num_bits
        errors      = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
        error_rate  = errors / sifted_len if sifted_len > 0 else 0

        print(f"  Raw bits sent         : {num_bits}")
        print(f"  Sifted key length     : {sifted_len}")
        print(f"  Key rate              : {key_rate:.1%}")
        print(f"  Bit errors (no Eve)   : {errors} ({error_rate:.1%})")
        print(f"  Alice key (sifted)    : {''.join(map(str, sifted_alice[:16]))}...")
        print(f"  Bob   key (sifted)    : {''.join(map(str, sifted_bob[:16]))}...")
        print(f"\n  With eavesdropper (Eve), error rate rises to ~25%,")
        print(f"  making interception detectable. This is the quantum advantage:")
        print(f"  any measurement disturbs the quantum state, revealing Eve.")

        return {
            'alice_key': sifted_alice,
            'bob_key': sifted_bob,
            'sifted_key': sifted_alice,
            'key_rate': key_rate,
            'error_rate': error_rate,
            'eavesdrop_detected': error_rate > 0.15,
        }


if __name__ == "__main__":
    qc = QuantumCryptanalysis()
    qc.grover_key_search_demo(key_bits=3)
    qc.shors_period_finding_demo(N=15, a=7)
    qc.bb84_qkd_simulation(num_bits=24)
