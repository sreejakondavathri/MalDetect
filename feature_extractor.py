import os
import re
import math

def calculate_entropy(data):
    """Calculate Shannon entropy of binary data"""
    if not data or len(data) == 0:
        return 0.0
    
    entropy = 0.0
    byte_counts = [0] * 256
    
    for byte in data:
        byte_counts[byte] += 1
    
    for count in byte_counts:
        if count > 0:
            p_x = float(count) / len(data)
            entropy += - p_x * math.log2(p_x)
    
    return round(entropy, 4)

def estimate_sections(data):
    """Estimate number of PE sections"""
    try:
        if len(data) < 200:
            return 0
        section_count = data[60:200].count(b'\x00\x00\x00\x00\x00\x00\x00\x00')
        return min(section_count, 10)
    except:
        return 0

def extract_features(filepath):
    """
    Extract comprehensive features from binary file
    Returns dictionary of numeric features in consistent order
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        entropy = calculate_entropy(data)
        
        # Suspicious API strings to search for
        suspicious_apis = [
            b'VirtualAlloc', b'VirtualProtect', b'CreateRemoteThread',
            b'WriteProcessMemory', b'LoadLibrary', b'GetProcAddress',
            b'WinExec', b'ShellExecute', b'URLDownloadToFile'
        ]
        
        # Count presence of each API
        api_counts = {api.decode(): int(api in data) for api in suspicious_apis}
        total_suspicious = sum(api_counts.values())
        
        # Check for PE signature (MZ header)
        has_pe_signature = int(data[:2] == b'MZ' and b'PE\x00\x00' in data[:1024])
        
        # Estimate number of sections
        num_sections = estimate_sections(data) if has_pe_signature else 0
        
        # Extract readable ASCII strings
        readable_strings = re.findall(b'[\x20-\x7E]{4,}', data)
        num_strings = len(readable_strings)
        avg_string_length = sum(len(s) for s in readable_strings) / max(len(readable_strings), 1)
        
        # Ensure consistent numeric features for both RF and LSTM
        features = {
            'file_size': float(file_size),
            'entropy': float(entropy),
            'num_sections': float(num_sections),
            'has_pe_signature': float(has_pe_signature),
            'num_strings': float(num_strings),
            'avg_string_length': round(float(avg_string_length), 2),
            'has_VirtualAlloc': float(api_counts.get('VirtualAlloc', 0)),
            'has_VirtualProtect': float(api_counts.get('VirtualProtect', 0)),
            'has_CreateRemoteThread': float(api_counts.get('CreateRemoteThread', 0)),
            'has_WriteProcessMemory': float(api_counts.get('WriteProcessMemory', 0)),
            'total_suspicious_apis': float(total_suspicious),
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

if __name__ == "__main__":
    # Test the feature extractor
    print("✅ Feature Extractor Module - Ready")
    print("Use it in your app: from feature_extractor import extract_features")
