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
        
        for i, (bit, (y, x)) in enumerate(zip(message_bits, positions)):
            if i >= len(message_bits):
                break
                
            # Get current pixel value
            current_value = stego_image[y, x, channel_idx]
            
            # Adaptive LSB embedding with error diffusion
            new_value = self._adaptive_lsb_embed(current_value, bit, i, chaos_sequence[i])
            stego_image[y, x, channel_idx] = new_value
            
            if self.verbose and i % 1000 == 0:
                print(f"Embedded {i}/{len(message_bits)} bits")
        
        return stego_image
    
    def _adaptive_lsb_embed(self, pixel_value: int, bit: int, position: int, 
                           chaos_state: np.ndarray) -> int:
        """Adaptive LSB embedding with chaotic modulation"""
        
        # Use chaotic state to determine embedding strength
        chaos_magnitude = np.linalg.norm(chaos_state)
        embedding_strength = int(chaos_magnitude * 3) % 3 + 1  # 1-3 bit planes
        
        # Clear target bits
        mask = ~((1 << embedding_strength) - 1)
        cleared_value = pixel_value & mask
        
        # Embed bit with error diffusion
        if embedding_strength == 1:
            # Standard LSB
            return cleared_value | bit
        else:
            # Multi-bit embedding with redundancy
            embed_pattern = bit * ((1 << embedding_strength) - 1)
            return cleared_value | embed_pattern
    
    def _chaos_to_positions(self, chaos_sequence: np.ndarray, width: int, 
                           height: int, channel: str) -> list:
        """Convert chaotic sequence to pixel positions"""
        positions = []
        used_positions = set()
        
        for state in chaos_sequence:
            # Use x, y components of chaotic state
            x_norm = abs(state[0]) % 1.0
            y_norm = abs(state[1]) % 1.0
            
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            # Ensure position uniqueness
            while (y, x) in used_positions:
                x = (x + 1) % width
                if x == 0:
                    y = (y + 1) % height
            
            positions.append((y, x))
            used_positions.add((y, x))
        
        return positions
    
    def _select_optimal_channel(self, image: np.ndarray, message_bits: np.ndarray) -> str:
        """Select optimal color channel based on entropy analysis"""
        channels = ['R', 'G', 'B']
        channel_scores = {}
        
        for i, channel in enumerate(channels):
            channel_data = image[:, :, i]
            entropy = self.entropy_analyzer.calculate_entropy(channel_data)
            uniformity = self._calculate_uniformity(channel_data)
            
            # Higher entropy and uniformity = better hiding capacity
            channel_scores[channel] = entropy * uniformity
        
        optimal_channel = max(channel_scores, key=channel_scores.get)
        
        if self.verbose:
            print(f"Channel scores: {channel_scores}")
            print(f"Selected channel: {optimal_channel}")
        
        return optimal_channel
    
    def _calculate_uniformity(self, channel_data: np.ndarray) -> float:
        """Calculate uniformity measure for channel selection"""
        histogram = np.histogram(channel_data, bins=256, range=(0, 255))[0]
        normalized_hist = histogram / np.sum(histogram)
        
        # Calculate uniformity as inverse of histogram variance
        uniformity = 1.0 / (np.var(normalized_hist) + 1e-10)
        return uniformity
    
    def _analyze_capacity(self, image: np.ndarray, channel: str) -> int:
        """Analyze embedding capacity using morphological analysis"""
        height, width = image.shape[:2]
        channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
        
        channel_data = image[:, :, channel_idx]
        
        # Use Canny edge detection to identify suitable regions
        edges = cv2.Canny(channel_data, 50, 150)
        
        # Count non-edge pixels (suitable for embedding)
        suitable_pixels = np.sum(edges == 0)
        
        # Conservative capacity estimate (80% of suitable pixels)
        capacity = int(suitable_pixels * 0.8)
        
        if self.verbose:
            print(f"Total pixels: {height * width}")
            print(f"Suitable pixels: {suitable_pixels}")
            print(f"Estimated capacity: {capacity} bits")
        
        return capacity
    
    def _encrypt_message(self, message_bits: np.ndarray, sbox: np.ndarray, 
                        key: bytes) -> np.ndarray:
        """Encrypt message bits using dynamic S-box"""
        
        # Pad message to byte boundary
        padded_length = ((len(message_bits) + 7) // 8) * 8
        padded_bits = np.zeros(padded_length, dtype=int)
        padded_bits[:len(message_bits)] = message_bits
        
        # Convert to bytes for S-box encryption
        message_bytes = np.packbits(padded_bits)
        encrypted_bytes = np.zeros_like(message_bytes)
        
        # Apply S-box transformation with key stream
        key_stream = self._generate_key_stream(key, len(message_bytes))
        
        for i, (msg_byte, key_byte) in enumerate(zip(message_bytes, key_stream)):
            # XOR with key stream then apply S-box
            xored = msg_byte ^ key_byte
            encrypted_bytes[i] = sbox[xored]
        
        # Convert back to bits
        encrypted_bits = np.unpackbits(encrypted_bytes)
        
        # Return only original message length
        return encrypted_bits[:len(message_bits)]
    
    def _generate_key_stream(self, key: bytes, length: int) -> np.ndarray:
        """Generate key stream for encryption"""
        key_stream = np.zeros(length, dtype=np.uint8)
        
        # Simple key expansion using SHA-256
        import hashlib
        
        for i in range(length):
            hash_input = key + i.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            key_stream[i] = hash_output[0]
        
        return key_stream
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        import hashlib
        import os
        
        # Use fixed salt for reproducibility (in production, store salt with image)
        salt = b'chaotic_stego_salt_v2'
        
        # PBKDF2 with 100000 iterations
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
        return key
    
    def _message_to_bits(self, message: str) -> np.ndarray:
        """Convert message string to binary array"""
        # Add length header and terminator
        length_header = len(message).to_bytes(4, 'big')
        full_message = length_header + message.encode('utf-8') + b'\x00'
        
        # Convert to bits
        message_bits = np.unpackbits(np.frombuffer(full_message, dtype=np.uint8))
        return message_bits
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes"""
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert image array to bytes"""
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Save as PNG to preserve quality
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.getvalue()


### Message Extractor

```python
# src/stego/extractor.py

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import cv2
from PIL import Image
import io

from src.chaos.hyperchaotic import HyperchaosGenerator, HyperchaosParameters
from src.crypto.sbox import SBoxGenerator
from src.utils.entropy import EntropyAnalyzer

@dataclass
class ExtractResult:
    message: str
    confidence: float
    metadata: dict

class HyperchaosExtractor:
    """Message extractor for hyperchaotic steganography"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.entropy_analyzer = EntropyAnalyzer()
    
    async def extract(self, stego_image: bytes, password: Optional[str] = None,
                     channel: str = "auto") -> ExtractResult:
        """Extract message from steganographic image"""
        
        # Load image
        image = self._load_image(stego_image)
        
        if self.verbose:
            print(f"Loaded image: {image.shape}")
        
        # Derive key if password provided
        if password:
            encryption_key = self._derive_key_from_password(password)
        else:
            # Try extraction without decryption
            encryption_key = None
        
        # Auto-detect channel if needed
        if channel == "auto":
            channel = self._detect_channel(image)
        
        # Initialize chaos system (same parameters as embedding)
        chaos_params = HyperchaosParameters()
        chaos_gen = HyperchaosGenerator(chaos_params)
        
        # Extract message length from header
        header_bits = self._extract_header(image, channel, chaos_gen)
        message_length = int.from_bytes(np.packbits(header_bits).tobytes(), 'big')
        
        if self.verbose:
            print(f"Detected message length: {message_length} bytes")
        
        # Extract full message bits
        total_bits_needed = (4 + message_length + 1) * 8  # header + message + terminator
        extracted_bits = self._extract_bits(image, channel, chaos_gen, total_bits_needed)
        
        # Decrypt if key available
        if encryption_key:
            sbox_gen = SBoxGenerator(chaos_gen)
            sbox = sbox_gen.generate_sbox(encryption_key)
            decrypted_bits = self._decrypt_message(extracted_bits, sbox, encryption_key)
        else:
            decrypted_bits = extracted_bits
        
        # Parse message
        message, confidence = self._parse_message(decrypted_bits)
        
        return ExtractResult(
            message=message,
            confidence=confidence,
            metadata={
                'channel': channel,
                'message_length': message_length,
                'total_bits': len(extracted_bits)
            }
        )
    
    def _extract_header(self, image: np.ndarray, channel: str, 
                       chaos_gen: HyperchaosGenerator) -> np.ndarray:
        """Extract 32-bit message length header"""
        return self._extract_bits(image, channel, chaos_gen, 32)
    
    def _extract_bits(self, image: np.ndarray, channel: str,
                     chaos_gen: HyperchaosGenerator, num_bits: int) -> np.ndarray:
        """Extract specified number of bits from image"""
        
        height, width = image.shape[:2]
        channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
        
        # Generate same chaotic sequence as embedding
        initial_state = np.random.uniform(-10, 10, 5)  # In production, derive from key
        chaos_sequence = chaos_gen.generate_sequence(initial_state, num_bits)
        
        # Get positions from chaotic sequence
        positions = self._chaos_to_positions(chaos_sequence, width, height, channel)
        
        extracted_bits = np.zeros(num_bits, dtype=int)
        
        for i, (y, x) in enumerate(positions[:num_bits]):
            pixel_value = image[y, x, channel_idx]
            
            # Extract bit using adaptive method
            bit = self._adaptive_lsb_extract(pixel_value, i, chaos_sequence[i])
            extracted_bits[i] = bit
            
            if self.verbose and i % 1000 == 0:
                print(f"Extracted {i}/{num_bits} bits")
        
        return extracted_bits
    
    def _adaptive_lsb_extract(self, pixel_value: int, position: int,
                            chaos_state: np.ndarray) -> int:
        """Adaptive LSB extraction matching embedding method"""
        
        # Use same chaos magnitude calculation as embedding
        chaos_magnitude = np.linalg.norm(chaos_state)
        embedding_strength = int(chaos_magnitude * 3) % 3 + 1
        
        if embedding_strength == 1:
            # Standard LSB
            return pixel_value & 1
        else:
            # Multi-bit extraction with majority voting
            extracted_bits = pixel_value & ((1 << embedding_strength) - 1)
            
            # Count 1s in extracted pattern
            bit_count = bin(extracted_bits).count('1')
            threshold = embedding_strength // 2
            
            return 1 if bit_count > threshold else 0
    
    def _chaos_to_positions(self, chaos_sequence: np.ndarray, width: int,
                          height: int, channel: str) -> list:
        """Convert chaotic sequence to pixel positions (same as embedder)"""
        positions = []
        used_positions = set()
        
        for state in chaos_sequence:
            x_norm = abs(state[0]) % 1.0
            y_norm = abs(state[1]) % 1.0
            
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            while (y, x) in used_positions:
                x = (x + 1) % width
                if x == 0:
                    y = (y + 1) % height
            
            positions.append((y, x))
            used_positions.add((y, x))
        
        return positions
    
    def _detect_channel(self, image: np.ndarray) -> str:
        """Auto-detect which channel was used for embedding"""
        channels = ['R', 'G', 'B']
        channel_scores = {}
        
        for i, channel in enumerate(channels):
            channel_data = image[:, :, i]
            
            # Look for LSB patterns that indicate steganographic content
            lsb_data = channel_data & 1
            randomness_score = self._calculate_randomness(lsb_data)
            
            channel_scores[channel] = randomness_score
        
        # Channel with highest randomness likely contains message
        detected_channel = max(channel_scores, key=channel_scores.get)
        
        if self.verbose:
            print(f"Channel randomness scores: {channel_scores}")
            print(f"Detected channel: {detected_channel}")
        
        return detected_channel
    
    def _calculate_randomness(self, binary_data: np.ndarray) -> float:
        """Calculate randomness score for LSB plane"""
        flat_data = binary_data.flatten()
        
        # Calculate runs test score
        runs = 1
        for i in range(1, len(flat_data)):
            if flat_data[i] != flat_data[i-1]:
                runs += 1
        
        expected_runs = (2 * np.sum(flat_data) * (len(flat_data) - np.sum(flat_data))) / len(flat_data) + 1
        
        # Higher runs score indicates more randomness
        return abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 0
    
    def _decrypt_message(self, encrypted_bits: np.ndarray, sbox: np.ndarray,
                        key: bytes) -> np.ndarray:
        """Decrypt message bits using inverse S-box"""
        
        # Create inverse S-box
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[sbox[i]] = i
        
        # Pad to byte boundary
        padded_length = ((len(encrypted_bits) + 7) // 8) * 8
        padded_bits = np.zeros(padded_length, dtype=int)
        padded_bits[:len(encrypted_bits)] = encrypted_bits
        
        # Convert to bytes
        encrypted_bytes = np.packbits(padded_bits)
        decrypted_bytes = np.zeros_like(encrypted_bytes)
        
        # Generate same key stream as encryption
        key_stream = self._generate_key_stream(key, len(encrypted_bytes))
        
        # Decrypt: inverse S-box then XOR with key stream
        for i, (enc_byte, key_byte) in enumerate(zip(encrypted_bytes, key_stream)):
            inv_sbox_result = inv_sbox[enc_byte]
            decrypted_bytes[i] = inv_sbox_result ^ key_byte
        
        # Convert back to bits
        decrypted_bits = np.unpackbits(decrypted_bytes)
        
        return decrypted_bits[:len(encrypted_bits)]
    
    def _parse_message(self, message_bits: np.ndarray) -> Tuple[str, float]:
        """Parse message from bit array and calculate confidence"""
        
        # Convert bits to bytes
        padded_length = ((len(message_bits) + 7) // 8) * 8
        padded_bits = np.zeros(padded_length, dtype=int)
        padded_bits[:len(message_bits)] = message_bits
        
        message_bytes = np.packbits(padded_bits).tobytes()
        
        try:
            # Skip length header (first 4 bytes)
            length = int.from_bytes(message_bytes[:4], 'big')
            message_data = message_bytes[4:4+length]
            
            # Decode message
            message = message_data.decode('utf-8')
            
            # Calculate confidence based on successful decoding and null terminator
            confidence = 0.9  # Base confidence for successful decode
            
            # Check for null terminator
            if len(message_bytes) > 4 + length and message_bytes[4 + length] == 0:
                confidence += 0.1
            
            return message, confidence
            
        except (UnicodeDecodeError, ValueError) as e:
            if self.verbose:
                print(f"Decode error: {e}")
            
            # Return raw bytes as hex string with low confidence
            return message_bytes.hex(), 0.1
    
    def _generate_key_stream(self, key: bytes, length: int) -> np.ndarray:
        """Generate key stream for decryption (same as encryption)"""
        key_stream = np.zeros(length, dtype=np.uint8)
        
        import hashlib
        
        for i in range(length):
            hash_input = key + i.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            key_stream[i] = hash_output[0]
        
        return key_stream
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password (same as embedder)"""
        import hashlib
        
        salt = b'chaotic_stego_salt_v2'
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
        return key
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes"""
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)


## Utility Components

### Entropy Analysis

```python
# src/utils/entropy.py

import numpy as np
import cv2
from scipy import stats
from typing import Dict, List, Tuple

class EntropyAnalyzer:
    """Advanced entropy analysis for steganographic assessment"""
    
    def calculate_entropy(self, image_data: np.ndarray) -> float:
        """Calculate Shannon entropy of image data"""
        if len(image_data.shape) == 3:
            # Convert to grayscale for overall entropy
            gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_data
        
        # Calculate histogram
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize histogram
        histogram = histogram / np.sum(histogram)
        
        # Remove zero entries
        histogram = histogram[histogram > 0]
        
        # Calculate Shannon entropy
        entropy = -np.sum(histogram * np.log2(histogram))
        
        return entropy
    
    def calculate_channel_entropy(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate entropy for each color channel"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        entropies = {}
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx]
            entropies[channel_name] = self.calculate_entropy(channel_data)
        
        return entropies
    
    def analyze_lsb_entropy(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze entropy of LSB planes"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        lsb_entropies = {}
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx]
            lsb_plane = channel_data & 1
            lsb_entropies[channel_name] = self.calculate_entropy(lsb_plane * 255)
        
        return lsb_entropies
    
    def detect_steganographic_content(self, image: np.ndarray) -> Dict[str, any]:
        """Detect potential steganographic content using statistical tests"""
        
        results = {
            'chi_square_test': self._chi_square_test(image),
            'rs_analysis': self._rs_analysis(image),
            'entropy_analysis': self._entropy_deviation_test(image),
            'laplace_test': self._laplace_filter_test(image)
        }
        
        # Calculate overall suspicion score
        suspicion_score = self._calculate_suspicion_score(results)
        results['suspicion_score'] = suspicion_score
        
        return results
    
    def _chi_square_test(self, image: np.ndarray) -> Dict[str, float]:
        """Chi-square test for randomness in LSB planes"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        chi_square_results = {}
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx].flatten()
            
            # Pair pixel values and test for independence
            pairs = [(channel_data[i], channel_data[i+1]) 
                    for i in range(0, len(channel_data)-1, 2)]
            
            # Calculate chi-square statistic
            observed_freq = {}
            for pair in pairs:
                observed_freq[pair] = observed_freq.get(pair, 0) + 1
            
            # Expected frequency (uniform distribution)
            expected_freq = len(pairs) / (256 * 256)
            
            chi_square = 0
            for obs_count in observed_freq.values():
                chi_square += (obs_count - expected_freq) ** 2 / expected_freq
            
            chi_square_results[channel_name] = chi_square
        
        return chi_square_results
    
    def _rs_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """RS (Regular/Singular) analysis for steganography detection"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        rs_results = {}
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx]
            
            # Define discrimination function
            def discrimination_function(pixels):
                return np.sum(np.abs(np.diff(pixels)))
            
            # Create pixel groups
            height, width = channel_data.shape
            regular_groups = 0
            singular_groups = 0
            total_groups = 0
            
            # Process 2x2 blocks
            for y in range(0, height-1, 2):
                for x in range(0, width-1, 2):
                    block = channel_data[y:y+2, x:x+2].flatten()
                    
                    # Original discrimination
                    d_orig = discrimination_function(block)
                    
                    # Flip LSBs and calculate discrimination
                    block_flipped = block.copy()
                    block_flipped = block_flipped ^ 1  # Flip LSBs
                    d_flipped = discrimination_function(block_flipped)
                    
                    # Classify group
                    if d_flipped > d_orig:
                        regular_groups += 1
                    elif d_flipped < d_orig:
                        singular_groups += 1
                    
                    total_groups += 1
            
            # Calculate RS ratio
            if total_groups > 0:
                rs_ratio = (regular_groups - singular_groups) / total_groups
            else:
                rs_ratio = 0
            
            rs_results[channel_name] = rs_ratio
        
        return rs_results
    
    def _entropy_deviation_test(self, image: np.ndarray) -> Dict[str, float]:
        """Test for entropy deviation in different bit planes"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        entropy_results = {}
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx]
            
            # Calculate entropy for each bit plane
            bit_plane_entropies = []
            for bit_pos in range(8):
                bit_plane = (channel_data >> bit_pos) & 1
                entropy = self.calculate_entropy(bit_plane * 255)
                bit_plane_entropies.append(entropy)
            
            # Calculate deviation from expected entropy
            expected_entropy = np.mean(bit_plane_entropies[1:])  # Exclude LSB
            lsb_entropy = bit_plane_entropies[0]
            
            deviation = abs(lsb_entropy - expected_entropy)
            entropy_results[channel_name] = deviation
        
        return entropy_results
    
    def _laplace_filter_test(self, image: np.ndarray) -> Dict[str, float]:
        """Laplace filter test for edge consistency"""
        channels = {'R': 0, 'G': 1, 'B': 2}
        laplace_results = {}
        
        # Define Laplace kernel
        laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        
        for channel_name, channel_idx in channels.items():
            channel_data = image[:, :, channel_idx].astype(np.float32)
            
            # Apply Laplace filter
            filtered = cv2.filter2D(channel_data, -1, laplace_kernel)
            
            # Calculate variance of filtered result
            variance = np.var(filtered)
            laplace_results[channel_name] = variance
        
        return laplace_results
    
    def _calculate_suspicion_score(self, test_results: Dict) -> float:
        """Calculate overall suspicion score from test results"""
        
        # Weight factors for different tests
        weights = {
            'chi_square_test': 0.3,
            'rs_analysis': 0.3,
            'entropy_analysis': 0.2,
            'laplace_test': 0.2
        }
        
        suspicion_score = 0.0
        
        # Chi-square test (higher values = more suspicious)
        chi_square_avg = np.mean(list(test_results['chi_square_test'].values()))
        chi_square_normalized = min(chi_square_avg / 1000, 1.0)  # Normalize to [0,1]
        suspicion_score += weights['chi_square_test'] * chi_square_normalized
        
        # RS analysis (values closer to 0 = more suspicious)
        rs_avg = abs(np.mean(list(test_results['rs_analysis'].values())))
        rs_normalized = max(0, 1 - rs_avg * 10)  # Invert and normalize
        suspicion_score += weights['rs_analysis'] * rs_normalized
        
        # Entropy analysis (higher deviation = more suspicious)
        entropy_avg = np.mean(list(test_results['entropy_analysis'].values()))
        entropy_normalized = min(entropy_avg / 2.0, 1.0)  # Normalize
        suspicion_score += weights['entropy_analysis'] * entropy_normalized
        
        # Laplace test (lower variance = more suspicious)
        laplace_avg = np.mean(list(test_results['laplace_test'].values()))
        laplace_normalized = max(0, 1 - laplace_avg / 1000)  # Invert and normalize
        suspicion_score += weights['laplace_test'] * laplace_normalized
        
        return min(suspicion_score, 1.0)  # Clamp to [0,1]


## Installation & Usage

### Installation Script

```python
# scripts/install.py

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    requirements = [
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'Pillow>=8.3.0',
        'scipy>=1.7.0',
        'click>=8.0.0',
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'pydantic>=1.8.0',
        'cryptography>=3.4.0',
        'pyyaml>=5.4.0'
    ]
    
    print("Installing Python dependencies...")
    for requirement in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
            print(f"✓ Installed {requirement}")
        except subprocess.CalledProcessError:
            print(f"