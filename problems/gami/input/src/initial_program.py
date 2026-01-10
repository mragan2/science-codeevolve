def solve(data_stream):
    """
    Goal: Identify a 12-bit signal (0b111000111000) through noise.
    """
    # EVOLVE-BLOCK-START
    # Improved strategy using Hamming distance for error correction within 2 bits
    target_signal = 0b111000111000
    
    # Precomputed lookup table for all 12-bit values within Hamming distance <= 2 from target
    # This includes the target itself and all values that differ by 1 or 2 bits
    valid_signals = {
        3640, 3632, 3636, 3644, 3648, 3656, 3672, 3608, 3592, 3560,
        3576, 3704, 3768, 3896, 3384, 3128, 2616, 4152, 4664, 5688,
        3633, 3635, 3637, 3639, 3641, 3643, 3645, 3647, 3649, 3651,
        3653, 3655, 3657, 3659, 3661, 3663, 3665, 3667, 3669, 3671,
        3673, 3675, 3677, 3679, 3681, 3683, 3685, 3687, 3689, 3691,
        3693, 3695, 3697, 3699, 3701, 3703, 3705, 3707, 3709, 3711,
        # ... (complete set would include all 79 values at distance <=2)
        # For brevity, we'll use a dynamic computation instead
    }
    
    # Dynamic generation of valid signals (more flexible than static table)
    def generate_nearby_signals(signal, max_distance=2, width=12):
        """Generate all signals within max_distance Hamming distance."""
        from collections import deque
        queue = deque([(signal, 0)])  # (current_signal, current_distance)
        visited = {signal}
        result = {signal}
        
        while queue:
            current_sig, dist = queue.popleft()
            if dist < max_distance:
                # Try flipping each bit
                for i in range(width):
                    flipped = current_sig ^ (1 << i)
                    if flipped not in visited:
                        visited.add(flipped)
                        result.add(flipped)
                        queue.append((flipped, dist + 1))
        return result
    
    # Generate valid signals dynamically
    valid_signals = generate_nearby_signals(target_signal)
    
    # Check if input matches any valid signal
    return 1 if data_stream in valid_signals else 0
    # EVOLVE-BLOCK-END
