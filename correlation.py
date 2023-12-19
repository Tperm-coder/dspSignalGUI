def correlation(signal1, signal2):
    # Ensure that signals have the same length
    assert len(signal1) == len(signal2), "Input signals must have the same length"

    # Calculate the summation squares for signal 1
    sum_squares_x = np.sum(np.square(signal1))

    # Initialize lists to store the indices and correlation values for each circular shift
    indices = []
    correlations = []
    N = len(signal1)

    # Calculate correlation for each circular shift
    for shift in range(N):
        # Circular shift the second signal manually
        shifted_signal2 = signal2[shift:] + signal2[:shift]
        # Calculate the summation squares for the circular shifted signal 2
        sum_squares_y = np.sum(np.square(shifted_signal2))
        # Calculate p for the circular shifted signal 2
        p = np.sqrt(sum_squares_x * sum_squares_y) / N
        # Calculate r for the circular shifted signal 2
        r = np.sum(np.multiply(signal1, shifted_signal2))
        # Divide by N
        r /= N
        # Calculate correlation and append to the lists
        corr = r / p
        correlations.append(corr)
        indices.append(shift)

    return indices, correlations