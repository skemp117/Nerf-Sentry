import multiprocessing
import time

# Function for the first process
def process1(process2_queue):
    while True:
        # Simulate some time-consuming work
        time.sleep(3)

        # Generate two data values to pass to process2
        data1 = "Data 1 from process1"
        data2 = "Data 2 from process1"
        data = (data1, data2)  # Create a tuple to hold both data values

        # Pass the data to process2
        process2_queue.put(data)

# Function for the second process
def process2(process2_queue):
    while True:
        data = process2_queue.get()  # Block until data is available
        data1, data2 = data  # Unpack the tuple
        print(f"Received data in process2 - Data 1: {data1}, Data 2: {data2}")

if __name__ == "__main__":
    # Create a multiprocessing queue to pass data between processes
    process2_queue = multiprocessing.Queue()

    # Create two multiprocessing processes for each process
    p1 = multiprocessing.Process(target=process1, args=(process2_queue,))
    p2 = multiprocessing.Process(target=process2, args=(process2_queue,))

    # Start the processes
    p1.start()
    p2.start()

    # Wait for the processes to finish (you can add a termination condition)
    p1.join()
    p2.join()
