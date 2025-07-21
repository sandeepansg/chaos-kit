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