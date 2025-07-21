# Chaos-Based Steganography System
## Project Architecture Documentation

### Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Design](#architecture-design)
3. [Enhanced Features](#enhanced-features)
4. [API Modules](#api-modules)
5. [Implementation Examples](#implementation-examples)
6. [Installation & Usage](#installation--usage)
7. [Security Analysis](#security-analysis)

---

## Project Overview

This project introduces a chaos-based steganography system that uses hyperchaotic encryption and bitplane manipulation to hide messages within digital images. The proposed architecture emphasizes modularity, API-driven design, and enhanced cryptographic security.

### Key Improvements
- **5D Hyperchaotic System** for enhanced randomness
- **Dynamic S-Box Generation** with cryptographic validation
- **Feistel Network Block Cipher** with CBC mode
- **Neural Network Authentication** using Hopfield networks
- **Adaptive Parameter Optimization** with Shark Smell Optimization
- **Modern Python Architecture** (3.10+) with type hints and async support

---

## Architecture Design

### Core System Architecture
```
ChaosSteg/
├── core/
│   ├── __init__.py
│   ├── hyperchaos.py          # 5D Hyperchaotic System
│   ├── crypto.py              # Cryptographic Primitives
│   ├── steganography.py       # Core Stego Operations
│   └── neural.py              # Hopfield Networks
├── algorithms/
│   ├── __init__.py
│   ├── sbox.py                # Dynamic S-Box Generation
│   ├── feistel.py             # Feistel Network Cipher
│   ├── chebyshev.py           # Polynomial Key Exchange
│   ├── chaos_hash.py          # Pure Chaos Hashing
│   └── optimization.py       # Shark Smell Optimization
├── utils/
│   ├── __init__.py
│   ├── image_ops.py           # Image Processing
│   ├── validators.py          # Cryptographic Validation
│   ├── prime_gen.py           # Enhanced Prime Generation
│   └── config.py              # Configuration Management
├── api/
│   ├── __init__.py
│   ├── rest_api.py            # FastAPI REST Endpoints
│   ├── cli.py                 # Command Line Interface
│   └── models.py              # Pydantic Data Models
├── tests/
├── docs/
├── scripts/
│   ├── setup.sh               # Linux/Mac Setup
│   └── setup.bat              # Windows Setup
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Enhanced Features

### 1. 5D Hyperchaotic System

#### Mathematical Model
```python
# Enhanced 5D Hyperchaotic System
dx/dt = 10(y - x) + u + ε₁(t)
dy/dt = 28x - y - x(w²) - v + ε₂(t)
dw/dt = k₁xyw - k₂w + k₃v + ε₃(t)
du/dt = -x(w²) + 2u + ε₄(t)
dv/dt = 8y + ε₅(t)
```

#### Enhanced Parameters
```python
HYPERCHAOS_PARAMS = {
    'k1': 1.0,      # coupling strength
    'k2': 4.0,      # damping coefficient  
    'k3': 1.2,      # cross-coupling
    'dt': 0.001,    # integration step
    'bounds': [-10, 10],  # state clipping
    'warmup': 1000,       # stabilization iterations
}
```

### 2. Dynamic S-Box Generation

#### Cryptographic Properties Validation
- **Bijection Property**: Each input maps to unique output
- **Nonlinearity**: High resistance to linear cryptanalysis
- **SAC (Strict Avalanche Criterion)**: Single bit change affects ≥50% output bits
- **BIC (Bit Independence Criterion)**: Output bit independence

### 3. Enhanced Security Features

#### Feistel Network with CBC Mode
```python
# Feistel Round Function
F(R, K) = P-Box(S-Box(R ⊕ K))

# CBC Mode Encryption
C₀ = IV
Cᵢ = E_K(Pᵢ ⊕ Cᵢ₋₁)
```

#### Hopfield Neural Network Authentication
```python
# Energy Function
E = -½ΣᵢΣⱼ wᵢⱼsᵢsⱼ - Σᵢθᵢsᵢ

# Update Rule
sᵢ(t+1) = sign(Σⱼwᵢⱼsⱼ(t) - θᵢ)
```

---

## API Modules

### 1. Core Hyperchaos Module

```python
# core/hyperchaos.py

from typing import Tuple, List, Optional
import numpy as np
from dataclasses import dataclass
import asyncio

@dataclass
class HyperchaosConfig:
    """Configuration for 5D Hyperchaotic System"""
    k1: float = 1.0
    k2: float = 4.0  
    k3: float = 1.2
    dt: float = 0.001
    bounds: Tuple[float, float] = (-10.0, 10.0)
    warmup_iterations: int = 1000

class HyperchaosGenerator:
    """Enhanced 5D Hyperchaotic Random Number Generator"""
    
    def __init__(self, config: HyperchaosConfig, initial_state: List[float]):
        self.config = config
        self.state = np.array(initial_state, dtype=np.float64)
        self.initialized = False
        
    def _clip_state(self) -> None:
        """Clip state values to prevent overflow"""
        min_val, max_val = self.config.bounds
        self.state = np.clip(self.state, min_val, max_val)
    
    def _rk4_step(self, noise: Optional[np.ndarray] = None) -> None:
        """Fourth-order Runge-Kutta integration step"""
        if noise is None:
            noise = np.zeros(5)
            
        def derivatives(state: np.ndarray) -> np.ndarray:
            x, y, w, u, v = state
            dxdt = 10 * (y - x) + u + noise[0]
            dydt = 28 * x - y - x * (w**2) - v + noise[1] 
            dwdt = self.config.k1 * x * y * w - self.config.k2 * w + self.config.k3 * v + noise[2]
            dudt = -x * (w**2) + 2 * u + noise[3]
            dvdt = 8 * y + noise[4]
            return np.array([dxdt, dydt, dwdt, dudt, dvdt])
        
        k1 = self.config.dt * derivatives(self.state)
        k2 = self.config.dt * derivatives(self.state + k1/2)
        k3 = self.config.dt * derivatives(self.state + k2/2)
        k4 = self.config.dt * derivatives(self.state + k3)
        
        self.state += (k1 + 2*k2 + 2*k3 + k4) / 6
        self._clip_state()
    
    def warmup(self) -> None:
        """Stabilize the chaotic system"""
        for _ in range(self.config.warmup_iterations):
            self._rk4_step()
        self.initialized = True
    
    def generate_sequence(self, length: int, quantization_bits: int = 8) -> np.ndarray:
        """Generate quantized chaotic sequence"""
        if not self.initialized:
            self.warmup()
            
        sequence = []
        max_val = 2**quantization_bits - 1
        
        for _ in range(length):
            self._rk4_step()
            # Use all state variables for better randomness
            combined = np.sum(np.abs(self.state))
            quantized = int((combined % 1.0) * max_val)
            sequence.append(quantized)
            
        return np.array(sequence, dtype=np.uint8)
    
    async def generate_sequence_async(self, length: int, batch_size: int = 1000) -> np.ndarray:
        """Async generation for large sequences"""
        sequence = []
        
        for i in range(0, length, batch_size):
            current_batch = min(batch_size, length - i)
            batch = self.generate_sequence(current_batch)
            sequence.extend(batch)
            
            if i % (batch_size * 10) == 0:
                await asyncio.sleep(0)  # Allow other tasks to run
                
        return np.array(sequence, dtype=np.uint8)

# Example Usage
def example_hyperchaos():
    """Example hyperchaos usage"""
    config = HyperchaosConfig()
    initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    generator = HyperchaosGenerator(config, initial_state)
    random_sequence = generator.generate_sequence(1000, quantization_bits=8)
    
    print(f"Generated sequence length: {len(random_sequence)}")
    print(f"Sequence entropy: {calculate_entropy(random_sequence):.4f}")
    
    return random_sequence
```

### 2. Dynamic S-Box Generation

```python
# algorithms/sbox.py

import numpy as np
from typing import List, Dict
from .crypto_validation import validate_sbox

class DynamicSBox:
    """Dynamic S-Box generator using hyperchaotic sequences"""
    
    def __init__(self, hyperchaos_generator):
        self.generator = hyperchaos_generator
        self.current_sbox = None
        self.inverse_sbox = None
        
    def generate_sbox(self, size: int = 256) -> np.ndarray:
        """Generate cryptographically strong S-Box"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate chaotic sequence
            chaotic_seq = self.generator.generate_sequence(size * 2)
            
            # Create bijective mapping
            sbox = self._create_bijective_mapping(chaotic_seq, size)
            
            # Validate cryptographic properties
            if self._validate_sbox(sbox):
                self.current_sbox = sbox
                self.inverse_sbox = self._compute_inverse(sbox)
                return sbox
                
        raise ValueError(f"Failed to generate valid S-Box after {max_attempts} attempts")
    
    def _create_bijective_mapping(self, sequence: np.ndarray, size: int) -> np.ndarray:
        """Create bijective S-Box from chaotic sequence"""
        # Fisher-Yates shuffle using chaotic sequence
        sbox = np.arange(size)
        
        for i in range(size - 1, 0, -1):
            j = sequence[i] % (i + 1)
            sbox[i], sbox[j] = sbox[j], sbox[i]
            
        return sbox
    
    def _validate_sbox(self, sbox: np.ndarray) -> bool:
        """Validate S-Box cryptographic properties"""
        return (
            self._check_bijection(sbox) and
            self._check_nonlinearity(sbox) >= 100 and  # Minimum nonlinearity
            self._check_sac(sbox) >= 0.45 and          # SAC criterion
            self._check_bic(sbox) >= 0.45              # BIC criterion
        )
    
    def _check_bijection(self, sbox: np.ndarray) -> bool:
        """Check if S-Box is bijective"""
        return len(set(sbox)) == len(sbox)
    
    def _check_nonlinearity(self, sbox: np.ndarray) -> float:
        """Calculate Walsh-Hadamard nonlinearity"""
        n = len(sbox)
        max_correlation = 0
        
        for a in range(1, n):
            for b in range(n):
                correlation = 0
                for x in range(n):
                    correlation += (-1) ** (bin(a & x).count('1') + bin(b & sbox[x]).count('1'))
                max_correlation = max(max_correlation, abs(correlation))
        
        return (n - max_correlation) / 2
    
    def _check_sac(self, sbox: np.ndarray) -> float:
        """Check Strict Avalanche Criterion"""
        n = len(sbox)
        bit_length = n.bit_length() - 1
        dependencies = 0
        total_tests = 0
        
        for i in range(n):
            for bit_pos in range(bit_length):
                flipped = i ^ (1 << bit_pos)
                if flipped < n:
                    diff = sbox[i] ^ sbox[flipped]
                    dependencies += bin(diff).count('1')
                    total_tests += bit_length
        
        return dependencies / total_tests if total_tests > 0 else 0
    
    def _check_bic(self, sbox: np.ndarray) -> float:
        """Check Bit Independence Criterion"""
        n = len(sbox)
        bit_length = n.bit_length() - 1
        independence_sum = 0
        total_pairs = 0
        
        for i in range(bit_length):
            for j in range(i + 1, bit_length):
                correlation = 0
                for x in range(n):
                    bit_i = (sbox[x] >> i) & 1
                    bit_j = (sbox[x] >> j) & 1
                    correlation += (-1) ** (bit_i ^ bit_j)
                independence_sum += abs(correlation)
                total_pairs += 1
        
        return 1 - (independence_sum / (total_pairs * n)) if total_pairs > 0 else 0
    
    def _compute_inverse(self, sbox: np.ndarray) -> np.ndarray:
        """Compute inverse S-Box"""
        inverse = np.zeros(len(sbox), dtype=sbox.dtype)
        for i, val in enumerate(sbox):
            inverse[val] = i
        return inverse

# Example Usage
def example_sbox_generation():
    """Example S-Box generation"""
    from core.hyperchaos import HyperchaosGenerator, HyperchaosConfig
    
    config = HyperchaosConfig()
    generator = HyperchaosGenerator(config, [0.1, 0.2, 0.3, 0.4, 0.5])
    
    sbox_gen = DynamicSBox(generator)
    sbox = sbox_gen.generate_sbox()
    
    print(f"Generated S-Box: {sbox[:16]}...")  # First 16 values
    print(f"Nonlinearity: {sbox_gen._check_nonlinearity(sbox)}")
    print(f"SAC: {sbox_gen._check_sac(sbox):.4f}")
    print(f"BIC: {sbox_gen._check_bic(sbox):.4f}")
    
    return sbox
```

### 3. Feistel Network Block Cipher

```python
# algorithms/feistel.py

import numpy as np
from typing import List, Tuple
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES
import hashlib

class FeistelCipher:
    """Feistel Network Block Cipher with CBC Mode"""
    
    def __init__(self, sbox_generator, rounds: int = 16, block_size: int = 128):
        self.sbox_gen = sbox_generator
        self.rounds = rounds
        self.block_size = block_size  # bits
        self.block_bytes = block_size // 8
        
    def _generate_round_keys(self, master_key: bytes) -> List[bytes]:
        """Generate round keys from master key"""
        round_keys = []
        current_key = master_key
        
        for round_num in range(self.rounds):
            # Hash current key with round number
            hasher = hashlib.sha256()
            hasher.update(current_key)
            hasher.update(round_num.to_bytes(4, 'big'))
            round_key = hasher.digest()[:self.block_bytes // 2]
            round_keys.append(round_key)
            current_key = round_key
            
        return round_keys
    
    def _feistel_function(self, right_half: bytes, round_key: bytes) -> bytes:
        """Feistel round function F(R, K)"""
        # XOR with round key
        xor_result = bytes(a ^ b for a, b in zip(right_half, round_key))
        
        # Apply S-Box substitution
        sbox = self.sbox_gen.current_sbox
        if sbox is None:
            sbox = self.sbox_gen.generate_sbox()
            
        substituted = bytes(sbox[b] for b in xor_result)
        
        # Apply permutation (simple bit rotation)
        permuted = self._permute_bytes(substituted)
        
        return permuted
    
    def _permute_bytes(self, data: bytes) -> bytes:
        """Apply permutation to bytes"""
        bit_string = ''.join(format(byte, '08b') for byte in data)
        
        # Simple permutation: rotate bits
        rotated = bit_string[1:] + bit_string[0]
        
        # Convert back to bytes
        result = []
        for i in range(0, len(rotated), 8):
            byte_bits = rotated[i:i+8]
            result.append(int(byte_bits, 2))
            
        return bytes(result)
    
    def encrypt_block(self, plaintext_block: bytes, round_keys: List[bytes]) -> bytes:
        """Encrypt single block using Feistel network"""
        if len(plaintext_block) != self.block_bytes:
            raise ValueError(f"Block size must be {self.block_bytes} bytes")
            
        half_size = self.block_bytes // 2
        left = plaintext_block[:half_size]
        right = plaintext_block[half_size:]
        
        # Feistel rounds
        for round_num in range(self.rounds):
            new_left = right
            f_output = self._feistel_function(right, round_keys[round_num])
            new_right = bytes(a ^ b for a, b in zip(left, f_output))
            
            left, right = new_left, new_right
            
        # Swap halves for final output
        return right + left
    
    def decrypt_block(self, ciphertext_block: bytes, round_keys: List[bytes]) -> bytes:
        """Decrypt single block using Feistel network"""
        if len(ciphertext_block) != self.block_bytes:
            raise ValueError(f"Block size must be {self.block_bytes} bytes")
            
        half_size = self.block_bytes // 2
        right = ciphertext_block[:half_size]  # Note: swapped for decryption
        left = ciphertext_block[half_size:]
        
        # Feistel rounds in reverse
        for round_num in range(self.rounds - 1, -1, -1):
            new_right = left
            f_output = self._feistel_function(left, round_keys[round_num])
            new_left = bytes(a ^ b for a, b in zip(right, f_output))
            
            left, right = new_left, new_right
            
        return left + right
    
    def encrypt_cbc(self, plaintext: bytes, key: bytes, iv: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypt using CBC mode with PKCS#7 padding"""
        if iv is None:
            iv = np.random.bytes(self.block_bytes)
            
        # Apply PKCS#7 padding
        padded_plaintext = pad(plaintext, self.block_bytes)
        
        # Generate round keys
        round_keys = self._generate_round_keys(key)
        
        ciphertext = b''
        previous_block = iv
        
        # Process each block
        for i in range(0, len(padded_plaintext), self.block_bytes):
            block = padded_plaintext[i:i + self.block_bytes]
            
            # XOR with previous ciphertext block (CBC mode)
            xor_block = bytes(a ^ b for a, b in zip(block, previous_block))
            
            # Encrypt block
            encrypted_block = self.encrypt_block(xor_block, round_keys)
            ciphertext += encrypted_block
            
            previous_block = encrypted_block
            
        return ciphertext, iv
    
    def decrypt_cbc(self, ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt using CBC mode"""
        if len(ciphertext) % self.block_bytes != 0:
            raise ValueError("Ciphertext length must be multiple of block size")
            
        # Generate round keys
        round_keys = self._generate_round_keys(key)
        
        plaintext = b''
        previous_block = iv
        
        # Process each block
        for i in range(0, len(ciphertext), self.block_bytes):
            block = ciphertext[i:i + self.block_bytes]
            
            # Decrypt block
            decrypted_block = self.decrypt_block(block, round_keys)
            
            # XOR with previous ciphertext block (CBC mode)
            xor_block = bytes(a ^ b for a, b in zip(decrypted_block, previous_block))
            plaintext += xor_block
            
            previous_block = block
            
        # Remove PKCS#7 padding
        return unpad(plaintext, self.block_bytes)

# Example Usage
def example_feistel_cipher():
    """Example Feistel cipher usage"""
    from algorithms.sbox import DynamicSBox
    from core.hyperchaos import HyperchaosGenerator, HyperchaosConfig
    
    # Setup
    config = HyperchaosConfig()
    generator = HyperchaosGenerator(config, [0.1, 0.2, 0.3, 0.4, 0.5])
    sbox_gen = DynamicSBox(generator)
    
    cipher = FeistelCipher(sbox_gen, rounds=16, block_size=128)
    
    # Test data
    plaintext = b"This is a secret message that needs to be encrypted!"
    key = b"secretkey1234567"  # 16 bytes for 128-bit key
    
    # Encrypt
    ciphertext, iv = cipher.encrypt_cbc(plaintext, key)
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext.hex()}")
    print(f"IV: {iv.hex()}")
    
    # Decrypt
    decrypted = cipher.decrypt_cbc(ciphertext, key, iv)
    print(f"Decrypted: {decrypted}")
    
    return ciphertext, iv
```

### 4. Command Line Interface

```python
# api/cli.py

import click
import asyncio
from pathlib import Path
from typing import Optional
import json

from core.steganography import ModernSteganography
from utils.config import load_config, save_config

@click.group()
@click.version_option()
def cli():
    """Modern Chaos-Based Steganography System"""
    pass

@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Input image file')
@click.option('--message', '-m', required=True, type=click.Path(exists=True),
              help='Message file to hide')
@click.option('--output', '-o', type=click.Path(),
              help='Output image file (default: auto-generated)')
@click.option('--password', '-p', required=True,
              help='Encryption password')
@click.option('--channel', '-c', type=click.Choice(['R', 'G', 'B']), default='R',
              help='Color channel to use')
@click.option('--plane', type=click.IntRange(0, 7), default=0,
              help='Bit plane to use')
@click.option('--config', type=click.Path(),
              help='Configuration file')
@click.option('--async-mode', is_flag=True,
              help='Use async processing for large files')
def embed(image: str, message: str, output: Optional[str], password: str,
          channel: str, plane: int, config: Optional[str], async_mode: bool):
    """Embed a message into an image using chaos-based steganography"""
    
    try:
        # Load configuration
        if config:
            cfg = load_config(config)
        else:
            cfg = load_config()
        
        # Initialize steganography system
        stego = ModernSteganography(cfg)
        
        # Generate output filename if not provided
        if not output:
            image_path = Path(image)
            output = str(image_path.parent / f"{image_path.stem}_embedded{image_path.suffix}")
        
        click.echo(f"Embedding message from '{message}' into '{image}'...")
        click.echo(f"Using channel: {channel}, bit plane: {plane}")
        
        if async_mode:
            result = asyncio.run(stego.embed_async(
                image_path=image,
                message_path=message,
                output_path=output,
                password=password,
                channel=channel,
                bit_plane=plane
            ))
        else:
            result = stego.embed(
                image_path=image,
                message_path=message,
                output_path=output,
                password=password,
                channel=channel,
                bit_plane=plane
            )
        
        if result.success:
            click.echo(f"✅ Message successfully embedded!")
            click.echo(f"Output image: {output}")
            click.echo(f"Capacity used: {result.capacity_used:.2%}")
            click.echo(f"Entropy change: {result.entropy_change:.6f}")
        else:
            click.echo(f"❌ Embedding failed: {result.error}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Input image file with embedded message')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for extracted message')
@click.option('--password', '-p', required=True,
              help='Decryption password')
@click.option('--channel', '-c', type=click.Choice(['R', 'G', 'B']), default='R',
              help='Color channel to use')
@click.option('--plane', type=click.IntRange(0, 7), default=0,
              help='Bit plane to use')
@click.option('--config', type=click.Path(),
              help='Configuration file')
def extract(image: str, output: Optional[str], password: str,
           channel: str, plane: int, config: Optional[str]):
    """Extract a hidden message from an image"""
    
    try:
        # Load configuration
        if config:
            cfg = load_config(config)
        else:
            cfg = load_config()
        
        # Initialize steganography system
        stego = ModernSteganography(cfg)
        
        # Generate output filename if not provided
        if not output:
            image_path = Path(image)
            output = str(image_path.parent / f"{image_path.stem}_extracted.txt")
        
        click.echo(f"Extracting message from '{image}'...")
        click.echo(f"Using channel: {channel}, bit plane: {plane}")
        
        result = stego.extract(
            image_path=image,
            output_path=output,
            password=password,
            channel=channel,
            bit_plane=plane
        )
        
        if result.success:
            click.echo(f"✅ Message successfully extracted!")
            click.echo(f"Output file: {output}")
            click.echo(f"Message length: {result.message_length} bytes")
            
            # Preview message if it's text
            if result.message_preview:
                click.echo(f"Preview: {result.message_preview[:100]}...")
        else:
            click.echo(f"❌ Extraction failed: {result.error}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

@cli.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Image file to analyze')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for analysis results')
@click.option('--detailed', is_flag=True,
              help='Generate detailed analysis reports')
def analyze(image: str, output: Optional[str], detailed: bool):
    """Analyze an image for steganographic content"""
    
    try:
        from utils.image_analyzer import ImageAnalyzer
        
        analyzer = ImageAnalyzer()
        
        click.echo(f"Analyzing image: {image}")
        
        result = analyzer.analyze_image(
            image_path=image,
            detailed=detailed
        )
        
        # Display basic results
        click.echo(f"\n📊 Analysis Results:")
        click.echo(f"Image size: {result.width}x{result.height}")
        click.echo(f"Channels: {result.channels}")
        click.echo(f"Entropy (R/G/B): {result.entropy}")
        click.echo(f"Potential capacity: {result.capacity} bytes")
        
        if result.steganography_probability > 0.5:
            click.echo(f"⚠️  Possible steganographic content detected!")
            click.echo(f"Probability: {result.steganography_probability:.2%}")
        else:
            click.echo(f"✅ No obvious steganographic content detected")
            
        # Save detailed results if requested
        if detailed and output:
            output_path = Path(output)
            output_path.mkdir(exist_ok=True)
            
            analyzer.save_analysis_report(result, output_path)
            click.echo(f"Detailed analysis saved to: {output}")
            
    except Exception as e:
        # Issue here
		click.echo(f"❌ Error: {str(e)}", err=True)

@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.json',
              help='Output configuration file')
def generate_config(output: str):
    """Generate default configuration file"""
    
    try:
        from utils.config import create_default_config
        
        config = create_default_config()
        save_config(config, output)
        
        click.echo(f"✅ Default configuration saved to: {output}")
        click.echo("Edit the configuration file to customize parameters.")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

@cli.command()
@click.option('--length', '-l', type=int, default=1000,
              help='Length of test sequence')
@click.option('--tests', type=click.Choice(['all', 'entropy', 'chi2', 'runs']), 
              default='all', help='Randomness tests to run')
def test_chaos(length: int, tests: str):
    """Test hyperchaotic random number generator"""
    
    try:
        from core.hyperchaos import HyperchaosGenerator, HyperchaosConfig
        from utils.randomness_tests import RandomnessTestSuite
        
        click.echo(f"Testing hyperchaotic generator with {length} samples...")
        
        # Generate test sequence
        config = HyperchaosConfig()
        generator = HyperchaosGenerator(config, [0.1, 0.2, 0.3, 0.4, 0.5])
        sequence = generator.generate_sequence(length)
        
        # Run tests
        test_suite = RandomnessTestSuite()
        results = test_suite.run_tests(sequence, test_types=tests)
        
        # Display results
        click.echo("\n🧪 Randomness Test Results:")
        for test_name, result in results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            click.echo(f"{test_name}: {status} (p-value: {result['p_value']:.6f})")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

if __name__ == '__main__':
    cli()
```

## Core Steganography Module

```python
# core/steganography.py

import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import hashlib
import struct
import asyncio
from pathlib import Path

from .hyperchaos import HyperchaosGenerator, HyperchaosConfig
from algorithms.sbox import DynamicSBox
from algorithms.feistel import FeistelCipher
from algorithms.chaos_hash import ChaosHash
from core.neural import HopfieldAuth
from utils.image_ops import ImageProcessor

@dataclass
class EmbedResult:
    """Result of embedding operation"""
    success: bool
    capacity_used: float
    entropy_change: float
    error: Optional[str] = None

@dataclass
class ExtractResult:
    """Result of extraction operation"""
    success: bool
    message_length: int
    message_preview: Optional[str] = None
    error: Optional[str] = None

class ModernSteganography:
    """Modern chaos-based steganography system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.image_processor = ImageProcessor()
        self.chaos_hash = ChaosHash()
        
        # Initialize hyperchaotic system
        self.hyperchaos_config = HyperchaosConfig(**config.get('hyperchaos', {}))
        
    def _derive_initial_conditions(self, password: str) -> list:
        """Derive initial conditions from password"""
        hash_bytes = hashlib.sha256(password.encode()).digest()
        
        # Convert to 5 floating point values in range [-1, 1]
        conditions = []
        for i in range(5):
            byte_val = hash_bytes[i * 4:(i + 1) * 4]
            int_val = struct.unpack('>I', byte_val)[0]
            float_val = (int_val / 2**32) * 2 - 1  # Scale to [-1, 1]
            conditions.append(float_val)
            
        return conditions
    
    def _prepare_message(self, message_data: bytes) -> bytes:
        """Prepare message with length header and checksum"""
        # Add message length (4 bytes)
        length_header = struct.pack('>I', len(message_data))
        
        # Add checksum (4 bytes)
        checksum = struct.pack('>I', self.chaos_hash.hash_bytes(message_data))
        
        # Combine header + checksum + message
        return length_header + checksum + message_data
    
    def _extract_message_info(self, data: bytes) -> Tuple[int, int, bytes]:
        """Extract message length, checksum, and data from prepared message"""
        if len(data) < 8:
            raise ValueError("Insufficient data for message header")
            
        length = struct.unpack('>I', data[:4])[0]
        checksum = struct.unpack('>I', data[4:8])[0]
        message_data = data[8:8 + length]
        
        return length, checksum, message_data
    
    def _embed_into_bitplane(self, image_array: np.ndarray, message_bits: np.ndarray,
                           channel: int, bit_plane: int, positions: np.ndarray) -> np.ndarray:
        """Embed message bits into specified bit plane"""
        modified_image = image_array.copy()
        
        for i, pos in enumerate(positions[:len(message_bits)]):
            row, col = divmod(pos, image_array.shape[1])
            if row < image_array.shape[0]:
                # Clear the target bit
                modified_image[row, col, channel] &= ~(1 << bit_plane)
                # Set the message bit
                modified_image[row, col, channel] |= (message_bits[i] << bit_plane)
                
        return modified_image
    
    def _extract_from_bitplane(self, image_array: np.ndarray, num_bits: int,
                             channel: int, bit_plane: int, positions: np.ndarray) -> np.ndarray:
        """Extract bits from specified bit plane"""
        extracted_bits = []
        
        for i, pos in enumerate(positions[:num_bits]):
            row, col = divmod(pos, image_array.shape[1])
            if row < image_array.shape[0]:
                bit = (image_array[row, col, channel] >> bit_plane) & 1
                extracted_bits.append(bit)
                
        return np.array(extracted_bits, dtype=np.uint8)
    
    def embed(self, image_path: str, message_path: str, output_path: str,
              password: str, channel: str = 'R', bit_plane: int = 0) -> EmbedResult:
        """Embed message into image"""
        
        try:
            # Load image and message
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            with open(message_path, 'rb') as f:
                message_data = f.read()
            
            # Prepare message with header
            prepared_message = self._prepare_message(message_data)
            
            # Convert to bits
            message_bits = np.unpackbits(np.frombuffer(prepared_message, dtype=np.uint8))
            
            # Check capacity
            channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
            max_capacity = image_array.shape[0] * image_array.shape[1]
            
            if len(message_bits) > max_capacity:
                return EmbedResult(
                    success=False,
                    capacity_used=0,
                    entropy_change=0,
                    error=f"Message too large. Need {len(message_bits)} bits, have {max_capacity}"
                )
            
            # Initialize chaos system
            initial_conditions = self._derive_initial_conditions(password)
            generator = HyperchaosGenerator(self.hyperchaos_config, initial_conditions)
            
            # Generate embedding positions
            position_sequence = generator.generate_sequence(len(message_bits))
            max_pos = image_array.shape[0] * image_array.shape[1] - 1
            positions = position_sequence % max_pos
            
            # Calculate original entropy
            original_entropy = self.image_processor.calculate_entropy(image_array[:,:,channel_idx])
            
            # Embed message
            modified_image = self._embed_into_bitplane(
                image_array, message_bits, channel_idx, bit_plane, positions
            )
            
            # Calculate new entropy
            new_entropy = self.image_processor.calculate_entropy(modified_image[:,:,channel_idx])
            
            # Save result
            result_image = Image.fromarray(modified_image)
            result_image.save(output_path)
            
            return EmbedResult(
                success=True,
                capacity_used=len(message_bits) / max_capacity,
                entropy_change=abs(new_entropy - original_entropy)
            )
            
        except Exception as e:
            return EmbedResult(
                success=False,
                capacity_used=0,
                entropy_change=0,
                error=str(e)
            )
    
    def extract(self, image_path: str, output_path: str, password: str,
                channel: str = 'R', bit_plane: int = 0) -> ExtractResult:
        """Extract message from image"""
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Initialize chaos system with same parameters
            initial_conditions = self._derive_initial_conditions(password)
            generator = HyperchaosGenerator(self.hyperchaos_config, initial_conditions)
            
            channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
            
            # Extract header (8 bytes = 64 bits)
            header_sequence = generator.generate_sequence(64)
            max_pos = image_array.shape[0] * image_array.shape[1] - 1
            header_positions = header_sequence % max_pos
            
            header_bits = self._extract_from_bitplane(
                image_array, 64, channel_idx, bit_plane, header_positions
            )
            
            # Reconstruct header bytes
            header_bytes = np.packbits(header_bits).tobytes()
            message_length, expected_checksum, _ = self._extract_message_info(header_bytes)
            
            # Extract message data
            total_bits_needed = 64 + (message_length * 8)  # header + message
            remaining_sequence = generator.generate_sequence(message_length * 8)
            message_positions = remaining_sequence % max_pos
            
            message_bits = self._extract_from_bitplane(
                image_array, message_length * 8, channel_idx, bit_plane, message_positions
            )
            
            # Reconstruct message
            message_bytes = np.packbits(message_bits).tobytes()
            
            # Verify checksum
            actual_checksum = self.chaos_hash.hash_bytes(message_bytes)
            if actual_checksum != expected_checksum:
                return ExtractResult(
                    success=False,
                    message_length=0,
                    error="Checksum verification failed - wrong password or corrupted data"
                )
            
            # Save extracted message
            with open(output_path, 'wb') as f:
                f.write(message_bytes)
            
            # Generate preview if text
            preview = None
            try:
                preview = message_bytes.decode('utf-8', errors='ignore')
            except:
                preview = f"Binary data ({len(message_bytes)} bytes)"
            
            return ExtractResult(
                success=True,
                message_length=len(message_bytes),
                message_preview=preview
            )
            
        except Exception as e:
            return ExtractResult(
                success=False,
                message_length=0,
                error=str(e)
            )
    
    async def embed_async(self, image_path: str, message_path: str, output_path: str,
                         password: str, channel: str = 'R', bit_plane: int = 0) -> EmbedResult:
        """Async version of embed for large files"""
        
        # For very large operations, process in chunks
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embed, image_path, message_path, output_path, password, channel, bit_plane
        )
```

## Neural Network Authentication

```python
# core/neural.py

import numpy as np
from typing import List, Tuple, Optional
import hashlib

class HopfieldAuth:
    """Hopfield Neural Network for Authentication"""
    
    def __init__(self, network_size: int = 256):
        self.network_size = network_size
        self.weights = None
        self.threshold = None
        self.trained_pattern = None
        
    def _string_to_pattern(self, data: str) -> np.ndarray:
        """Convert string to bipolar pattern (+1, -1)"""
        # Hash to fixed size
        hash_bytes = hashlib.sha256(data.encode()).digest()
        
        # Extend or truncate to network size
        pattern_bytes = (hash_bytes * ((self.network_size // 256) + 1))[:self.network_size]
        
        # Convert to bipolar
        pattern = np.array([1 if b > 127 else -1 for b in pattern_bytes])
        return pattern
    
    def train(self, auth_data: str) -> None:
        """Train network with authentication pattern"""
        pattern = self._string_to_pattern(auth_data)
        self.trained_pattern = pattern
        
        # Hebbian learning rule
        self.weights = np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
        self.threshold = np.zeros(self.network_size)
        
    def authenticate(self, test_data: str, max_iterations: int = 100,
                    tolerance: float = 0.9) -> Tuple[bool, float]:
        """Authenticate using Hopfield network convergence"""
        if self.weights is None:
            raise ValueError("Network not trained")
            
        test_pattern = self._string_to_pattern(test_data)
        current_state = test_pattern.copy()
        
        # Hopfield dynamics
        for iteration in range(max_iterations):
            # Asynchronous update
            for i in range(self.network_size):
                net_input = np.dot(self.weights[i], current_state) - self.threshold[i]
                current_state[i] = 1 if net_input >= 0 else -1
            
            # Check convergence to trained pattern
            similarity = np.mean(current_state == self.trained_pattern)
            
            if similarity >= tolerance:
                return True, similarity
                
        # Final similarity check
        final_similarity = np.mean(current_state == self.trained_pattern)
        return final_similarity >= tolerance, final_similarity
    
    def get_energy(self, pattern: np.ndarray) -> float:
        """Calculate Hopfield energy"""
        if self.weights is None:
            return 0.0
            
        energy = -0.5 * np.dot(pattern, np.dot(self.weights, pattern))
        energy -= np.dot(self.threshold, pattern)
        return energy
```

## Chaos Hash Algorithm

```python
# algorithms/chaos_hash.py

import numpy as np
from typing import Union
import struct

class ChaosHash:
    """Pure chaos-based hash function using hyperchaotic maps"""
    
    def __init__(self, hash_size: int = 32):
        self.hash_size = hash_size  # bytes
        self.hash_bits = hash_size * 8
        
    def _logistic_map(self, x: float, r: float = 3.99999) -> float:
        """Chaotic logistic map"""
        return r * x * (1 - x)
    
    def _tent_map(self, x: float, mu: float = 1.99999) -> float:
        """Chaotic tent map"""
        if x < 0.5:
            return mu * x
        else:
            return mu * (1 - x)
    
    def _henon_map(self, x: float, y: float, a: float = 1.4, b: float = 0.3) -> tuple:
        """2D Henon chaotic map"""
        x_new = 1 - a * x**2 + y
        y_new = b * x
        return x_new, y_new
    
    def hash_bytes(self, data: bytes) -> int:
        """Generate chaos-based hash of byte data"""
        if not data:
            return 0
            
        # Initialize chaos parameters from data
        seed = sum(data) % 1000000
        x1 = (seed % 997) / 997.0  # Avoid exact 0 or 1
        x2 = ((seed * 7919) % 991) / 991.0
        y1 = ((seed * 5881) % 983) / 983.0
        y2 = ((seed * 3571) % 977) / 977.0
        
        # Process each byte
        hash_state = np.zeros(self.hash_bits, dtype=np.uint8)
        
        for i, byte_val in enumerate(data):
            # Evolve chaotic systems
            x1 = self._logistic_map(x1)
            x2 = self._tent_map(x2)
            y1, y2 = self._henon_map(y1, y2)
            
            # Mix with input byte
            chaos_byte = int((x1 + x2 + y1 + y2) * 64) % 256
            mixed_byte = byte_val ^ chaos_byte
            
            # Update hash state
            bit_pos = (i * 8) % self.hash_bits
            for bit in range(8):
                if bit_pos + bit < self.hash_bits:
                    current_bit = (mixed_byte >> bit) & 1
                    hash_state[bit_pos + bit] ^= current_bit
        
        # Final chaos mixing
        for _ in range(self.hash_bits // 8):
            x1 = self._logistic_map(x1)
            x2 = self._tent_map(x2)
            y1, y2 = self._henon_map(y1, y2)
            
            mix_val = int((x1 + x2 + y1 + y2) * 64) % 256
            hash_state[_ % self.hash_bits] ^= mix_val & 1
        
        # Convert to integer
        hash_int = 0
        for i, bit in enumerate(hash_state[:32]):  # Use first 32 bits
            hash_int |= (bit << i)
            
        return hash_int
    
    def hash_string(self, data: str) -> str:
        """Generate hex hash of string data"""
        hash_int = self.hash_bytes(data.encode('utf-8'))
        return f"{hash_int:08x}"
```

## Image Processing Utilities

```python
# utils/image_ops.py

import numpy as np
from PIL import Image
from typing import Tuple, List
import cv2

class ImageProcessor:
    """Image processing utilities for steganography"""
    
    def calculate_entropy(self, image_data: np.ndarray) -> float:
        """Calculate Shannon entropy of image data"""
        # Get histogram
        hist, _ = np.histogram(image_data.flatten(), bins=256, range=(0, 256))
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def calculate_psnr(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(modified.shape) == 3:
            modified = cv2.cvtColor(modified, cv2.COLOR_RGB2GRAY)
            
        return cv2.SSIM(original, modified)[0]
    
    def get_embedding_capacity(self, image_shape: Tuple[int, ...], 
                             channels: List[str] = ['R'], 
                             bit_planes: List[int] = [0]) -> int:
        """Calculate embedding capacity in bits"""
        height, width = image_shape[:2]
        total_pixels = height * width
        
        capacity = total_pixels * len(channels) * len(bit_planes)
        return capacity
    
    def enhance_security(self, image_data: np.ndarray, 
                        noise_level: float = 0.1) -> np.ndarray:
        """Add subtle noise to enhance security"""
        noise = np.random.normal(0, noise_level, image_data.shape)
        enhanced = np.clip(image_data + noise, 0, 255).astype(np.uint8)
        return enhanced
```

## Configuration Management

```python
# utils/config.py

import json
from typing import Dict, Any
from pathlib import Path

DEFAULT_CONFIG = {
    "hyperchaos": {
        "k1": 1.0,
        "k2": 4.0,
        "k3": 1.2,
        "dt": 0.001,
        "bounds": [-10.0, 10.0],
        "warmup_iterations": 1000
    },
    "feistel": {
        "rounds": 16,
        "block_size": 128
    },
    "steganography": {
        "default_channel": "R",
        "default_bit_plane": 0,
        "max_capacity_ratio": 0.1
    },
    "security": {
        "enable_neural_auth": True,
        "enable_checksum": True,
        "noise_level": 0.05
    }
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        for section, values in user_config.items():
            if section in config:
                config[section].update(values)
            else:
                config[section] = values
                
        return config
    else:
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return DEFAULT_CONFIG.copy()
```

## Randomness Test Suite

```python
# utils/randomness_tests.py

import numpy as np
from scipy import stats
from typing import Dict, List, Union

class RandomnessTestSuite:
    """Statistical randomness tests for chaos sequences"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha  # Significance level
        
    def entropy_test(self, sequence: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Shannon entropy test"""
        unique, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Theoretical maximum entropy for 8-bit data
        max_entropy = 8.0
        normalized_entropy = entropy / max_entropy
        
        return {
            'passed': normalized_entropy > 0.95,
            'p_value': normalized_entropy,
            'entropy': entropy
        }
    
    def chi_square_test(self, sequence: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Chi-square goodness of fit test"""
        observed_freq = np.bincount(sequence, minlength=256)
        expected_freq = len(sequence) / 256
        
        chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=255)
        
        return {
            'passed': p_value > self.alpha,
            'p_value': p_value,
            'chi2_statistic': chi2_stat
        }
    
    def runs_test(self, sequence: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Runs test for randomness"""
        # Convert to binary based on median
        median = np.median(sequence)
        binary = (sequence > median).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        n1 = np.sum(binary == 1)
        n2 = np.sum(binary == 0)
        
        if n1 == 0 or n2 == 0:
            return {'passed': False, 'p_value': 0.0}
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        
        z_stat = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'passed': p_value > self.alpha,
            'p_value': p_value,
            'runs': runs,
            'expected_runs': expected_runs
        }
    
    def autocorrelation_test(self, sequence: np.ndarray, max_lag: int = 50) -> Dict[str, Union[bool, float]]:
        """Autocorrelation test"""
        autocorr = np.correlate(sequence - np.mean(sequence), 
                               sequence - np.mean(sequence), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Check if autocorrelation is significantly different from 0 for lags > 0
        significant_lags = np.sum(np.abs(autocorr[1:max_lag]) > 2/np.sqrt(len(sequence)))
        
        return {
            'passed': significant_lags < max_lag * 0.05,  # Less than 5% significant
            'p_value': 1 - (significant_lags / max_lag),
            'significant_lags': significant_lags
        }
    
    def run_tests(self, sequence: np.ndarray, 
                  test_types: str = 'all') -> Dict[str, Dict[str, Union[bool, float]]]:
        """Run selected randomness tests"""
        results = {}
        
        if test_types == 'all' or test_types == 'entropy':
            results['entropy'] = self.entropy_test(sequence)
            
        if test_types == 'all' or test_types == 'chi2':
            results['chi_square'] = self.chi_square_test(sequence)
            
        if test_types == 'all' or test_types == 'runs':
            results['runs'] = self.runs_test(sequence)
            
        if test_types == 'all':
            results['autocorrelation'] = self.autocorrelation_test(sequence)
        
        return results
```

## Installation Scripts

### setup.sh (Linux/Mac)
```bash
#!/bin/bash

echo "Setting up Modern Chaos-Based Steganography System..."

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Setup complete! Activate environment with: source venv/bin/activate"
echo "Run tests with: python -m pytest tests/"
echo "CLI help: python -m api.cli --help"
```

### setup.bat (Windows)
```batch
@echo off
echo Setting up Modern Chaos-Based Steganography System...

python --version
if %errorlevel% neq 0 (
    echo Error: Python 3.8+ required
    pause
    exit /b 1
)

python -m venv venv
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo Setup complete! Activate environment with: venv\Scripts\activate.bat
echo Run tests with: python -m pytest tests/
echo CLI help: python -m api.cli --help
pause
```

## Requirements.txt
```
numpy>=1.21.0
Pillow>=9.0.0
scipy>=1.7.0
opencv-python>=4.5.0
click>=8.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.0
cryptography>=3.4.0
pytest>=6.2.0
pytest-asyncio>=0.18.0
matplotlib>=3.5.0
```

## pyproject.toml
```toml
# pyproject.toml

[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "modern-chaos-steganography"
version = "2.0.0"
description = "Advanced chaos-based steganography system with hyperchaotic encryption"
authors = [{name = "Research Team", email = "research@example.com"}]
readme = "README.md"
license = {file = "LICENSE"}
keywords = ["steganography", "chaos", "cryptography", "security"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Security :: Cryptography",
    "Topic :: Scientific/Engineering"
]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "scipy>=1.10.0",
    "cryptography>=41.0.0",
    "click>=8.1.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0"
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.2.0"
]
ml = [
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "optuna>=3.3.0"
]

[project.scripts]
chaos-stego = "api.cli:cli"

[project.urls]
Homepage = "https://github.com/research/modern-chaos-steganography"
Documentation = "https://chaos-stego.readthedocs.io"
Repository = "https://github.com/research/modern-chaos-steganography"
Issues = "https://github.com/research/modern-chaos-steganography/issues"

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=core --cov-report=html --cov-report=term-missing"
```
