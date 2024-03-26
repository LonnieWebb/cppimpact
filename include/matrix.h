template <typename T, int MAX_SIZE, int MAX_BW>
class BandedMatrix
{
private:
    double data[MAX_SIZE * (2 * MAX_BW + 1)];

    // Method to compute the index in the 1D array
    int index(int row, int col) const
    {
        return row * (2 * MAX_BW + 1) + (col - row + MAX_BW);
    }

public:
    // Constructor
    BandedMatrix()
    {
        // Initialize the data array to zero
        for (int i = 0; i < MAX_SIZE * (2 * MAX_BW + 1); ++i)
        {
            data[i] = 0.0;
        }
    }

    // Method to set an element
    void set(int row, int col, double value)
    {
        if (abs(row - col) <= MAX_BW)
        {
            data[index(row, col)] = value;
        }
        // No error handling for simplicity; be careful with indices.
    }

    // Method to get an element
    double get(int row, int col) const
    {
        if (abs(row - col) <= MAX_BW)
        {
            return data[index(row, col)];
        }
        else
        {
            return 0.0; // Zero for elements outside the bandwidth
        }
    }
    void print() const
    {
        for (int row = 0; row < MAX_SIZE; ++row)
        {
            for (int col = 0; col < MAX_SIZE; ++col)
            {
                if (abs(row - col) <= MAX_BW)
                {
                    printf("%2.1f ", get(row, col));
                }
                else
                {
                    printf("0.0 ");
                }
            }
            printf("\n");
        }
    }
};
