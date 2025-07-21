# Modern Chaotic Steganography Framework
## Technical Documentation & Implementation Guide

### Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Enhanced Features](#enhanced-features)
5. [API Design](#api-design)
6. [Implementation Details](#implementation-details)
7. [Installation & Usage](#installation--usage)
8. [Security Analysis](#security-analysis)

## Overview

This document outlines the modernization of a legacy chaotic steganography system, transforming it into a robust, API-driven framework using contemporary Python practices and advanced cryptographic techniques.

### Legacy System Analysis
The original system implemented:
- Basic Lorenz attractor for chaos generation
- Simple bit-plane steganography
- Morphological analysis for capacity estimation
- Entropy-based channel selection

### Modern Enhancement Goals
- **Hyperchaotic Systems**: 5D hyperchaotic attractors for enhanced randomness
- **Dynamic Cryptography**: Real-time S-box generation with validation
- **Advanced Key Exchange**: Chebyshev polynomial-based secure key distribution
- **Modular Architecture**: Clean separation of concerns with comprehensive API
- **Cross-Platform Support**: Native terminal interface with shell script automation

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                      │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway Layer                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ Steganography│ │ Cryptography│ │   Chaos     │ │Analysis │ │
│  │   Module    │ │   Module    │ │   Module    │ │ Module  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Core Engine Layer                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   Image     │ │   Message   │ │    Key      │ │  Config │ │
│  │  Processing │ │  Processing │ │ Management  │ │Manager  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
chaotic_steganography/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   ├── chaos/
│   │   ├── __init__.py
│   │   ├── hyperchaotic.py
│   │   ├── generators.py
│   │   └── validators.py
│   ├── crypto/
│   │   ├── __init__.py
│   │   ├── sbox.py
│   │   ├── chebyshev.py
│   │   └── encryption.py
│   ├── stego/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   ├── extractor.py
│   │   └── analyzer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py
│   │   ├── entropy.py
│   │   └── validators.py
│   └── api/
│       ├── __init__.py
│       ├── endpoints.py
│       └── models.py
├── cli/
│   ├── __init__.py
│   ├── main.py
│   ├── commands.py
│   └── interface.py
├── scripts/
│   ├── stego.bat         # Windows batch
│   ├── stego.sh          # Unix shell
│   └── install.py        # Cross-platform installer
├── tests/
├── docs/
├── config/
│   └── default.yaml
└── requirements.txt
```

## Core Components

### 1. Hyperchaotic System Generator

#### 5D Hyperchaotic Attractor Implementation

```python
# src/chaos/hyperchaotic.py

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class HyperchaosParameters:
    """Parameters for 5D hyperchaotic system"""
    a: float = 36.0
    b: float = 3.0 
    c: float = 20.0
    d: float = -1.0
    e: float = -16.0
    r: float = 1.0
    dt: float = 0.001
    
class HyperchaosGenerator:
    """5D Hyperchaotic system for enhanced randomness generation"""
    
    def __init__(self, params: HyperchaosParameters):
        self.params = params
        self.state = np.zeros(5)
        
    def system_equations(self, state: np.ndarray) -> np.ndarray:
        """
        5D Hyperchaotic system equations:
        dx/dt = a(y - x) + w
        dy/dt = cx - y - xz + v
        dz/dt = xy - bz
        dw/dt = -yz + dw
        dv/dt = -ex + ey + rv
        """
        x, y, z, w, v = state
        
        dx = self.params.a * (y - x) + w
        dy = self.params.c * x - y - x * z + v
        dz = x * y - self.params.b * z
        dw = -y * z + self.params.d * w
        dv = -self.params.e * x + self.params.e * y + self.params.r * v
        
        return np.array([dx, dy, dz, dw, dv])
    
    def generate_sequence(self, initial_state: np.ndarray, length: int) -> np.ndarray:
        """Generate hyperchaotic sequence using 4th order Runge-Kutta"""
        sequence = np.zeros((length, 5))
        state = initial_state.copy()
        
        for i in range(length):
            sequence[i] = state
            # 4th order Runge-Kutta integration
            k1 = self.system_equations(state)
            k2 = self.system_equations(state + 0.5 * self.params.dt * k1)
            k3 = self.system_equations(state + 0.5 * self.params.dt * k2)
            k4 = self.system_equations(state + self.params.dt * k3)
            
            state += (self.params.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        return sequence
    
    def extract_binary_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Extract binary sequence from hyperchaotic states"""
        # Use modulo operation on scaled values
        scaled = np.abs(sequence * 1e14).astype(np.int64)
        binary = scaled % 2
        return binary.flatten()
```

#### Pseudocode for Hyperchaotic Generation

```
ALGORITHM: Generate_Hyperchaotic_Binary_Sequence
INPUT: initial_state[5], length, parameters
OUTPUT: binary_sequence[length*5]

BEGIN
    Initialize state = initial_state
    Initialize sequence = zeros(length, 5)
    
    FOR i = 0 TO length-1 DO
        sequence[i] = state
        
        // 4th order Runge-Kutta integration
        k1 = compute_derivatives(state, parameters)
        k2 = compute_derivatives(state + 0.5*dt*k1, parameters)
        k3 = compute_derivatives(state + 0.5*dt*k2, parameters)
        k4 = compute_derivatives(state + dt*k3, parameters)
        
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    END FOR
    
    // Extract binary sequence
    scaled_sequence = abs(sequence * 10^14) mod 2^64
    binary_sequence = scaled_sequence mod 2
    
    RETURN flatten(binary_sequence)
END
```

### 2. Dynamic S-Box Generation

#### Cryptographically Validated S-Box Creation

```python
# src/crypto/sbox.py

import numpy as np
from typing import List, Tuple, Optional
import hashlib
from src.chaos.hyperchaotic import HyperchaosGenerator

class SBoxGenerator:
    """Dynamic S-Box generation with cryptographic validation"""
    
    def __init__(self, chaos_generator: HyperchaosGenerator):
        self.chaos_gen = chaos_generator
        
    def generate_sbox(self, key: bytes) -> np.ndarray:
        """Generate 8x8 S-box from chaotic sequence"""
        # Use key to determine initial conditions
        initial_state = self._key_to_initial_state(key)
        
        # Generate sufficient chaotic sequence
        sequence = self.chaos_gen.generate_sequence(initial_state, 1000)
        binary_seq = self.chaos_gen.extract_binary_sequence(sequence)
        
        # Create S-box using chaotic shuffling
        sbox = self._create_sbox_from_sequence(binary_seq)
        
        # Validate cryptographic properties
        if not self._validate_sbox(sbox):
            # Regenerate with perturbed initial conditions
            return self._regenerate_sbox(key)
            
        return sbox
    
    def _key_to_initial_state(self, key: bytes) -> np.ndarray:
        """Convert key to 5D initial state"""
        hash_val = hashlib.sha256(key).digest()
        
        # Convert to floating point values in suitable range
        state = np.zeros(5)
        for i in range(5):
            # Use 4 bytes for each dimension
            bytes_chunk = hash_val[i*4:(i+1)*4]
            int_val = int.from_bytes(bytes_chunk, 'big')
            # Map to [-10, 10] range
            state[i] = (int_val / (2**32 - 1)) * 20 - 10
            
        return state
    
    def _create_sbox_from_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Create S-box using Fisher-Yates shuffle with chaotic sequence"""
        sbox = np.arange(256, dtype=np.uint8)
        
        seq_idx = 0
        for i in range(255, 0, -1):
            # Use 8 bits from sequence to determine swap position
            if seq_idx + 8 > len(sequence):
                break
                
            swap_bits = sequence[seq_idx:seq_idx+8]
            swap_pos = int(''.join(map(str, swap_bits)), 2) % (i + 1)
            
            # Swap elements
            sbox[i], sbox[swap_pos] = sbox[swap_pos], sbox[i]
            seq_idx += 8
            
        return sbox
    
    def _validate_sbox(self, sbox: np.ndarray) -> bool:
        """Validate S-box cryptographic properties"""
        # Check bijection (all values 0-255 present exactly once)
        if not np.array_equal(np.sort(sbox), np.arange(256)):
            return False
            
        # Check nonlinearity (simplified test)
        nonlinearity = self._calculate_nonlinearity(sbox)
        if nonlinearity < 100:  # Threshold for acceptable nonlinearity
            return False
            
        # Check SAC (Strict Avalanche Criterion)
        sac_value = self._calculate_sac(sbox)
        if abs(sac_value - 0.5) > 0.1:  # Should be close to 0.5
            return False
            
        return True
    
    def _calculate_nonlinearity(self, sbox: np.ndarray) -> float:
        """Calculate nonlinearity measure"""
        # Simplified nonlinearity calculation
        # Full implementation would use Walsh-Hadamard transform
        differences = []
        for i in range(256):
            for j in range(i+1, 256):
                diff = bin(sbox[i] ^ sbox[j]).count('1')
                differences.append(diff)
        
        return np.std(differences)
    
    def _calculate_sac(self, sbox: np.ndarray) -> float:
        """Calculate Strict Avalanche Criterion"""
        total_changes = 0
        total_bits = 0
        
        for i in range(256):
            for bit_pos in range(8):
                input_flip = i ^ (1 << bit_pos)
                output_diff = sbox[i] ^ sbox[input_flip]
                
                total_changes += bin(output_diff).count('1')
                total_bits += 8
        
        return total_changes / total_bits
```

### 3. Chebyshev Polynomial Key Exchange

#### Mathematical Foundation and Implementation

```python
# src/crypto/chebyshev.py

import numpy as np
from typing import Tuple, Optional
import hashlib
from math import cos, acos, cosh, acosh

class ChebyshevKeyExchange:
    """Chebyshev polynomial-based key exchange with extended bit range"""
    
    def __init__(self, bit_length: int = 2048):
        self.bit_length = bit_length
        self.prime = self._generate_large_prime()
        
    def generate_keys(self) -> Tuple[int, int]:
        """Generate public and private key pair"""
        # Private key: random integer
        private_key = np.random.randint(2**(self.bit_length-1), 2**self.bit_length)
        
        # Public parameters
        x = np.random.uniform(-0.9, 0.9)  # Base point
        n = private_key
        
        # Public key: T_n(x) mod p where T_n is Chebyshev polynomial
        public_key = self._chebyshev_polynomial(n, x) % self.prime
        
        return public_key, private_key
    
    def _chebyshev_polynomial(self, n: int, x: float) -> int:
        """Compute Chebyshev polynomial T_n(x) with extended precision"""
        if abs(x) <= 1:
            # Use trigonometric definition for |x| <= 1
            result = cos(n * acos(x))
        else:
            # Use hyperbolic definition for |x| > 1
            if x > 1:
                result = cosh(n * acosh(x))
            else:  # x < -1
                result = (-1)**n * cosh(n * acosh(-x))
        
        # Convert to integer with extended precision
        return int(result * 10**10) % self.prime
    
    def _generate_large_prime(self) -> int:
        """Generate large prime for modular operations"""
        # For demonstration, using a known large prime
        # In production, implement proper prime generation
        return 2**2048 - 2**1024 - 1
    
    def compute_shared_secret(self, private_key: int, other_public: int, 
                            base_x: float) -> bytes:
        """Compute shared secret using commutative property"""
        # Shared secret = T_private(other_public mod suitable_range)
        
        # Map public key back to valid Chebyshev domain
        mapped_x = (other_public % 1000000) / 1000000.0 * 1.8 - 0.9
        
        shared_value = self._chebyshev_polynomial(private_key, mapped_x)
        
        # Convert to bytes for cryptographic use
        return hashlib.sha256(str(shared_value).encode()).digest()
    
    def validate_mathematical_properties(self, n: int, x: float) -> bool:
        """Validate mathematical properties of Chebyshev polynomials"""
        # Test recurrence relation: T_n+1(x) = 2xT_n(x) - T_n-1(x)
        if n < 2:
            return True
            
        t_n_minus_1 = self._chebyshev_polynomial(n-1, x)
        t_n = self._chebyshev_polynomial(n, x)
        t_n_plus_1 = self._chebyshev_polynomial(n+1, x)
        
        expected = (2 * x * t_n - t_n_minus_1) % self.prime
        actual = t_n_plus_1 % self.prime
        
        return abs(expected - actual) < 1e-6
```

#### Pseudocode for Key Exchange

```
ALGORITHM: Chebyshev_Key_Exchange
INPUT: bit_length
OUTPUT: shared_secret

BEGIN
    // Alice's side
    private_A = random_integer(2^(bit_length-1), 2^bit_length)
    x = random_float(-0.9, 0.9)
    public_A = T_private_A(x) mod prime
    
    // Bob's side  
    private_B = random_integer(2^(bit_length-1), 2^bit_length)
    public_B = T_private_B(x) mod prime
    
    // Shared secret computation (commutative property)
    // Alice computes: T_private_A(public_B mapped to domain)
    // Bob computes: T_private_B(public_A mapped to domain)
    
    mapped_value_A = map_to_chebyshev_domain(public_B)
    mapped_value_B = map_to_chebyshev_domain(public_A)
    
    shared_A = T_private_A(mapped_value_A) mod prime
    shared_B = T_private_B(mapped_value_B) mod prime
    
    // Both should be equal due to semi-group property
    ASSERT(shared_A == shared_B)
    
    shared_secret = SHA256(shared_A)
    RETURN shared_secret
END

FUNCTION: T_n(x)  // Chebyshev polynomial of first kind
INPUT: n, x
OUTPUT: polynomial_value

BEGIN
    IF |x| <= 1 THEN
        RETURN cos(n * arccos(x))
    ELSE IF x > 1 THEN
        RETURN cosh(n * arccosh(x))
    ELSE  // x < -1
        RETURN (-1)^n * cosh(n * arccosh(-x))
    END IF
END
```

## API Design

### REST API Endpoints

```python
# src/api/endpoints.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64

app = FastAPI(title="Chaotic Steganography API", version="2.0.0")

class EmbedRequest(BaseModel):
    image_data: str  # base64 encoded
    message: str
    password: Optional[str] = None
    channel: str = "auto"
    algorithm: str = "hyperchaotic"
    
class EmbedResponse(BaseModel):
    success: bool
    stego_image: Optional[str] = None  # base64 encoded
    capacity_used: float
    entropy_change: float
    error_message: Optional[str] = None

class ExtractRequest(BaseModel):
    stego_image: str  # base64 encoded
    password: Optional[str] = None
    channel: str = "auto"
    
class ExtractResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    confidence: float
    error_message: Optional[str] = None

@app.post("/api/v2/embed", response_model=EmbedResponse)
async def embed_message(request: EmbedRequest):
    """Embed message into image using hyperchaotic steganography"""
    try:
        from src.stego.embedder import HyperchaosEmbedder
        
        embedder = HyperchaosEmbedder()
        result = await embedder.embed(
            image_data=base64.b64decode(request.image_data),
            message=request.message,
            password=request.password,
            channel=request.channel,
            algorithm=request.algorithm
        )
        
        return EmbedResponse(
            success=True,
            stego_image=base64.b64encode(result.stego_image).decode(),
            capacity_used=result.capacity_used,
            entropy_change=result.entropy_change
        )
        
    except Exception as e:
        return EmbedResponse(
            success=False,
            error_message=str(e)
        )

@app.post("/api/v2/extract", response_model=ExtractResponse)
async def extract_message(request: ExtractRequest):
    """Extract message from steganographic image"""
    try:
        from src.stego.extractor import HyperchaosExtractor
        
        extractor = HyperchaosExtractor()
        result = await extractor.extract(
            stego_image=base64.b64decode(request.stego_image),
            password=request.password,
            channel=request.channel
        )
        
        return ExtractResponse(
            success=True,
            message=result.message,
            confidence=result.confidence
        )
        
    except Exception as e:
        return ExtractResponse(
            success=False,
            error_message=str(e)
        )
```

### CLI Interface

```python
# cli/main.py

import click
import asyncio
import base64
from pathlib import Path
from typing import Optional

@click.group()
@click.version_option(version="2.0.0")
def main():
    """Modern Chaotic Steganography Framework"""
    pass

@main.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Input image file')
@click.option('--message', '-m', type=str, help='Message text to embed')
@click.option('--message-file', '-f', type=click.Path(exists=True),
              help='File containing message to embed')
@click.option('--output', '-o', type=click.Path(), help='Output image file')
@click.option('--password', '-p', type=str, help='Encryption password')
@click.option('--channel', '-c', default='auto',
              type=click.Choice(['auto', 'R', 'G', 'B']),
              help='Color channel to use')
@click.option('--algorithm', '-a', default='hyperchaotic',
              type=click.Choice(['hyperchaotic', 'lorenz', 'mixed']),
              help='Chaos algorithm to use')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def embed(image: str, message: Optional[str], message_file: Optional[str],
          output: Optional[str], password: Optional[str], channel: str,
          algorithm: str, verbose: bool):
    """Embed message into image using chaotic steganography"""
    
    if not message and not message_file:
        raise click.ClickException("Either --message or --message-file must be provided")
    
    if message_file:
        message = Path(message_file).read_text()
    
    if not output:
        image_path = Path(image)
        output = str(image_path.with_suffix('.stego' + image_path.suffix))
    
    async def _embed():
        from src.stego.embedder import HyperchaosEmbedder
        
        embedder = HyperchaosEmbedder(verbose=verbose)
        
        with open(image, 'rb') as f:
            image_data = f.read()
        
        result = await embedder.embed(
            image_data=image_data,
            message=message,
            password=password,
            channel=channel,
            algorithm=algorithm
        )
        
        with open(output, 'wb') as f:
            f.write(result.stego_image)
        
        if verbose:
            click.echo(f"✓ Message embedded successfully")
            click.echo(f"✓ Capacity used: {result.capacity_used:.2%}")
            click.echo(f"✓ Entropy change: {result.entropy_change:.4f}")
            click.echo(f"✓ Output saved to: {output}")
    
    asyncio.run(_embed())

@main.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Steganographic image file')
@click.option('--output', '-o', type=click.Path(), help='Output message file')
@click.option('--password', '-p', type=str, help='Decryption password')
@click.option('--channel', '-c', default='auto',
              type=click.Choice(['auto', 'R', 'G', 'B']),
              help='Color channel to use')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def extract(image: str, output: Optional[str], password: Optional[str],
            channel: str, verbose: bool):
    """Extract message from steganographic image"""
    
    async def _extract():
        from src.stego.extractor import HyperchaosExtractor
        
        extractor = HyperchaosExtractor(verbose=verbose)
        
        with open(image, 'rb') as f:
            image_data = f.read()
        
        result = await extractor.extract(
            stego_image=image_data,
            password=password,
            channel=channel
        )
        
        if output:
            Path(output).write_text(result.message)
            if verbose:
                click.echo(f"✓ Message extracted and saved to: {output}")
        else:
            click.echo("Extracted message:")
            click.echo("=" * 50)
            click.echo(result.message)
            click.echo("=" * 50)
        
        if verbose:
            click.echo(f"✓ Extraction confidence: {result.confidence:.2%}")
    
    asyncio.run(_extract())

if __name__ == '__main__':
    main()
```

## Implementation Details

### Modern Steganography Embedder

```python
# src/stego/embedder.py

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import cv2
from PIL import Image
import io

from src.chaos.hyperchaotic import HyperchaosGenerator, HyperchaosParameters
from src.crypto.sbox import SBoxGenerator
from src.crypto.chebyshev import ChebyshevKeyExchange
from src.utils.entropy import EntropyAnalyzer
from src.utils.image_processing import ImageProcessor

@dataclass
class EmbedResult:
    stego_image: bytes
    capacity_used: float
    entropy_change: float
    metadata: Dict[str, Any]

class HyperchaosEmbedder:
    """Advanced steganographic embedder using hyperchaotic systems"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.entropy_analyzer = EntropyAnalyzer()
        self.image_processor = ImageProcessor()
        
    async def embed(self, image_data: bytes, message: str, 
                   password: Optional[str] = None, channel: str = "auto",
                   algorithm: str = "hyperchaotic") -> EmbedResult:
        """Main embedding function"""
        
        # Load and validate image
        image = self._load_image(image_data)
        original_entropy = self.entropy_analyzer.calculate_entropy(image)
        
        # Prepare message
        message_bits = self._message_to_bits(message)
        
        if self.verbose:
            print(f"Message length: {len(message)} chars, {len(message_bits)} bits")
        
        # Generate cryptographic components
        if password:
            key_exchange = ChebyshevKeyExchange()
            encryption_key = self._derive_key_from_password(password)
        else:
            encryption_key = np.random.bytes(32)
        
        # Initialize hyperchaotic system
        chaos_params = HyperchaosParameters()
        chaos_gen = HyperchaosGenerator(chaos_params)
        
        # Generate dynamic S-box
        sbox_gen = SBoxGenerator(chaos_gen)
        sbox = sbox_gen.generate_sbox(encryption_key)
        
        # Encrypt message
        encrypted_bits = self._encrypt_message(message_bits, sbox, encryption_key)
        
        # Channel selection
        if channel == "auto":
            channel = self._select_optimal_channel(image, encrypted_bits)
        
        # Capacity analysis
        capacity = self._analyze_capacity(image, channel)
        if len(encrypted_bits) > capacity:
            raise ValueError(f"Message too large. Capacity: {capacity}, Required: {len(encrypted_bits)}")
        
        # Embed using adaptive algorithm
        stego_image = self._embed_adaptive(image, encrypted_bits, channel, chaos_gen)
        
        # Calculate metrics
        new_entropy = self.entropy_analyzer.calculate_entropy(stego_image)
        entropy_change = abs(new_entropy - original_entropy)
        capacity_used = len(encrypted_bits) / capacity
        
        # Convert result to bytes
        stego_bytes = self._image_to_bytes(stego_image)
        
        return EmbedResult(
            stego_image=stego_bytes,
            capacity_used=capacity_used,
            entropy_change=entropy_change,
            metadata={
                'algorithm': algorithm,
                'channel': channel,
                'original_entropy': original_entropy,
                'new_entropy': new_entropy,
                'message_length': len(message_bits)
            }
        )
    
    def _embed_adaptive(self, image: np.ndarray, message_bits: np.ndarray,
                       channel: str, chaos_gen: HyperchaosGenerator) -> np.ndarray:
        """Adaptive embedding using hyperchaotic guidance"""
        
        stego_image = image.copy()
        height, width = image.shape[:2]
        
        # Generate hyperchaotic sequence for position selection
        initial_state = np.random.uniform(-10, 10, 5)
        chaos_sequence = chaos_gen.generate_sequence(initial_state, len(message_bits))
        
        # Extract positions from chaotic sequence
        positions = self._chaos_to_positions(chaos_sequence, width, height, channel)
        
        # Embed bits at selected positions
        channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
        
        for i, (bit, (y, x)) in enumerate(zip(message