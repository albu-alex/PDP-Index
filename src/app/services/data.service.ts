import { Injectable } from '@angular/core';

export interface Exercise {
  id: number;
  name: string;
  index: string;
  question: string;
  answer: string;
  read: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class DataService {
  public exercises: Exercise[] = [
    {
      id: 0,
      name: 'Jan-Feb 2017 Subject No. 1',
      index: 'Problem 1',
      question: 'Write a parallel (distributed or local, at your choice) program for solving the c-coloring problem. That is, you are given a number k, and n objects and some pairs umong them that have to have distinct colors. Find n solution to color them with at most & colors in total, if one exist. Assume you have a function that gets a vector with k integers epresenting the assingment of colors to objects and checks if the constraints are obeyed or not.',
      answer: `
      #include <mpi.h>
      #include <vector>
      using namespace std;
      
      bool check_constraints(vector<int> colors) {
          // Check if the constraints are obeyed
          // ...
      }
      
      int main(int argc, char* argv[]) {
          int rank, size;
          MPI_Init(&argc, &argv);
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          MPI_Comm_size(MPI_COMM_WORLD, &size);
      
          int k = atoi(argv[1]);
          int n = atoi(argv[2]);
      
          // Divide the search space among processes
          int chunk_size = (int) pow(k, n) / size;
          int start = rank * chunk_size;
          int end = (rank + 1) * chunk_size;
      
          // Try all possible color assignments
          for (int i = start; i < end; i++) {
              vector<int> colors(n);
              int temp = i;
              for (int j = 0; j < n; j++) {
                  colors[j] = temp % k;
                  temp /= k;
              }
              if (check_constraints(colors)) {
                  // Send the solution to rank 0
                  MPI_Send(&colors[0], n, MPI_INT, 0, 0, MPI_COMM_WORLD);
                  break;
              }
          }
      
          if (rank == 0) {
              // Receive solutions from other processes
              vector<vector<int>> solutions;
              for (int i = 1; i < size; i++) {
                  vector<int> colors(n);
                  MPI_Recv(&colors[0], n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                  solutions.push_back(colors);
              }
              // Print the solutions
              for (auto colors : solutions) {
                  for (int color : colors) {
                      cout << color << " ";
                  }
                  cout << endl;
              }
          }
      
          MPI_Finalize();
          return 0;
      }
      `,
      read: false,
    },
    {
      id: 1,
      name: 'Jan-Feb 2017 Subject No. 1',
      index: 'Problem 2',
      question: `Consider the following code for inserting a new value into a linked list at a given position. We assume that insertions can be called concurrently, but not for the same position. Find and fix the concurrency issue. 
      Also, describe a function for parsing the linked list.`,
      answer: `
      struct Node {
        unsigned payload;
        Node next;
        Node prev:
        mutex atx;
      }

      void insertAfter (Node before, unsigned value) {
      Node* node = new Node;
      node->payload= value;
      Node* after =  before->next; 
      before->mtx.lock(); 
      before->next = node;
      before->mtx.unlock();
      after-mtx.lock();
      after->prev = node;
      after-mtx.unlock();
      node->prev  = before;
      node->next = after;

        Answer: 

        The concurrency issue in this code is that it does not properly synchronize access to the linked list. Specifically, there is a race condition between the two threads
        that are trying to insert new nodes into the list at different positions. One thread may be modifying the 'next' pointer of a node while another thread is trying to
        access the 'prev' pointer of that same node, resulting in undefined behavior.

        One way to fix this issue would be to use a global lock that is acquired before calling the insertAfter() function and released after the insertion is complete. This would
        ensure that only one thread can access the linked list at a time, preventing any race conditions.

        Another solution would be to use a lock-free data structure such as a lock-free linked list, which allows for concurrent access to the list without the need for locks.
        A function for parsing the linked list could look like this:

        void parseList(Node* head) {
          Node* current = head;
          while(current != nullptr) {
            cout << current->payload << endl;
            current = current-> next;
          }
        }

        This function takes the head of the linked list as an input and iterates through the list by following the 'next' pointers of each node. It prints the payload of each node
        as it goes along. This function is not thread-safe, so you need to make sure to synchronize the access to the linked list if the function is called concurrently.
      `,
      read: false
    },
    {
      id: 2,
      name: "Jan-Feb 2017 Subject No. 1",
      index: "Problem 3",
      question: `We have n servers that can communicate to each other. There are events producing on each of them; each event has an associated information. Each server must write a history of all events (server and event inio) produced on all servers, and, furthermore, the recorded history must be the same for all servers. Write a distributed algorithm that accomplishes that. Consider the case n = 2 for starting.`,
      answer: `
      One way to accomplish this would be to be use a distributed consensus algorithm such as Paxos or Raft. The basic idea is that each server would act as a leader in a round-robin
      fashion, and would propose new events to the other servers. The other servers would then vote on whether to accept the proposed events, and once a majority of servers have accepted
      the proposed events, they would be added to the history. We can implement this algorithm by using the MPI Library in C++.

      #include <mpi.h>
      #include <vector>

      struct Event {
        int server_id;
        std::string event_info;
      }

      std::vector<Event> event_history;

      void record_event(int server_id, std::string event_info) {
        Event new_event = {server_id, event_info};
        event_history.push_back(new_event);
      }

      int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int world_size, world_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        while(true) {
          // Wait for events to happen
          // ...

          // Record the event on the local history
          record_event(world_rank, "Event information");
          for (int i=0; i<world_size; i++) {
            if (i != world_rank) {
              MPI_Send(&world_rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
              MPI_Send("Event information", 17, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
          }

          // Recieve events from the other servers
          for (int i=0; i < world_size - 1; i++) {
            int server_id;
            MPI_Recv(&server_id, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char event_info[17];
            MPI_Recv(event_info, 17, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            record_event(server_id, event_info);
          }
        }
        MPI_Finalize();
      }

      This algorithm guarantees that all servers will have the same event history in the end. 
      It is important to notice that this algorithm is a simple solution for the case of n = 2, if the number of servers increases, it will not scale well, and other more robust algorithms should be applied.
      `,
      read: false
    },
    {
      id: 3,
      name: 'Jan-Feb 2017 Subject No.2',
      index: 'Problem 1',
      question: `Write a parallel (distributed or local, at your choice) program for finding a Hamiltonian path starting at a given vertex. 
      That is, you are given a graph with n vertices and must find a path that starts at vertex 0 and goes through each of the other vertices exactly once. Find a solution, if one exits. 
      If needed, assume you have a function that gets a vector containing a permutation of length n and verifies if it is Hamiltonian path in the given graph or not.`,
      answer: `
      #include <mpi.h>
      #include <vector>

      int n; // number of vertices
      std::vector<std::vector<int>> graph; // adjacency matrix representation of the graph
      std::vector<int> path // to store the found Hamiltonian path

      void search(int vertex, std::vector<bool> visited) {
        // base case: if all vertices have been visited
        if (path.size() == n) {
          // check if the path is Hamiltonian
          if (isHamiltonian(path)) {
            // print the found path
            for (int v : path) {
              printf("%d ", v);
            }
            printf("\n");
            MPI_Abort(MPI_COMM_WORLD, 1); // terminate all processes
          }
          return;
        }
        visted[vertex] = true;
        path.push_back(vertex);
        // recursively search for the next vertex
        for (int i =0; i < n; i++) {
          if (graph[vertex][i] && !visited[i]) {
            search(i, visited);
          }
        }
        visited[vertex] = false;
        path.pop_back();
      }

      int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Read the graph
        // ....

        std::vector<bool> visited(n, false);
        search(0, visited);

        MPI_Finalize();
        return 0;
      }

      We use a depth-first search to find a Hamiltonian path, starting from vertex 0. The function isHamiltonian is used to check if the path is valid, and it is already implemented.
      We also use MPI to parallelize the search process. Each process starts the search from a different vertex and perform the search in parallel. Once a Hamiltonian path
      is found, the search is terminated by calling 'MPI_Abort' to terminate all processes.
      `,
      read: false
    },
    {
      id: 4,
      name: 'Jan-Feb 2017 Subject No.2',
      index: 'Problem 2',
      question: 
      `
      Consider the following code for transferring money from one account to another. 
      You are required to write a function parsing all accounts (assume you have a vector <Account>) and compute the total amount of money there, so that it doesn't interfere with possible transfers at the same time. 
      Change the transfer function if needed, but it must be able to be called concurrently for independent pair of accounts.
      `,
      answer: 
      `
      struct Account {
        unsigned id;
        unsigned balance;
        mutex mtx;
      };
      
      bool transfer(Account& from, Account& to, unsigned amount) {
        {
          unique_lock<mutex> lck1(from.mtx);
          if(from.balance < amount) return false
          from.balance -= amount;
        }
        {
          unique_lock<mutex> lck2(to.mtx);
          to.balance += amount;
        }
      }

      Answer:

      The current implementation is using unique locks which allows only one thread to access the critical section at a time. However, this is not enough to ensure that the transfer
      function can be called concurrently for independent pairs of accounts, beacause both locks are on the same mutex. To avoid deadlocks, you need to lock the mutexes in a fixed
      order to ensure the same pair of accounts always locks their mutexes in the same order.

      Here's the updated code:

      bool transfer(Account& from, Account& to, unsigned amount) {
        Acount* accounts[] = {&from, &to};
        sort(accounts, accounts+2, [](const Account* a, const Account* b) { return a->id < b->id; });
        unique_lock<mutex> lck1(accounts[0]->mtx, defer_lock);
        unique_lock<mutex> lck2(accounts[1]->mtx, defer_lock);
        lock(lck1, lck2);
        if(from.balance < amount) return false;
        from.balance -= amount;
        to.balance += amount;
        return true;
      }

      The 'lock' function locks the two mutexes in the correct order and the 'defer_lock' option is used to prevent the lock from being acquired immediately, so that the lock order can be determined dynamically.
      `,
      read: false
    },
    {
      id: 5,
      name: "Jan-Feb 2017 Subject No. 2",
      index: "Problem 3",
      question: `We have n servers that can communicate to each other. There are events producing on each of them; each event has an associated information. Each server must write a history of all events (server and event inio) produced on all servers, and, furthermore, the recorded history must be the same for all servers. Write a distributed algorithm that accomplishes that. Consider the case n = 2 for starting.`,
      answer: `
      Solution
        Let's consider the case when n = 2. Let's name the processes A and B.

        Every process will keep a Lamport clock and pass that on each message.

        Suppose an event occurs in the process A. Then, process A will increate it's timestamp t by one, and send that event alongside t at the process B - this is called PREPARE event. Process B computes the maximum between its internal clock and the timestamp of the message and adds one. It send back an OK with that computed timestamp. Process A receives the OK and sends back a COMMIT message. They both agreed on this value.

        The generalization to n > 2 follows easily.

        Each process having an occuring event will:

        broadcast PREPARE with the timestamp
        waits for OKs from all the other processes
        computes the maximum amongs the OKs timestamps
        broadcast COMMIT to each process
        There is only one little problem. A process may have previously gave an OK but did not get any COMMIT yet for that event. So, in case another COMMIT appears, it can't write that to a file because the COMMIT from the previously sent OK may be either before, or after the current COMMIT. That's why, each process will maintain a list of given OKs as well as a list of COMMITs to be flushed to disk.

        Note. Every tie can be solved by chosing an initial arbitrary ordering of the processes. Such as the PID or IP if they live on different hosts.

        Side note: Also, check out the solution on the Subject No.1, because they have the exact same question.
      `,
      read: false
    },
    {
      id: 6,
      name: 'Feb 2018 Subject No.1',
      index: 'Problem 1',
      question: 
      `
      Write a parallel or distributed program that counts the number of permutations of N that satisfy a given property. 
      You have a function (bool pred(vector <int> const& v)) that verifies if a given permutation satisfies the property. 
      Your program shall call that function once for each permutation and count the number of times it returns true.
      `,
      answer: 
      `
      This problem is solved using a parallel program in C++.

      #include <iostream>
      #include <thread>
      #include <vector>
      #include <atomic>

      using namespace std;

      bool check(vector<int> v) {
        return (v[0] % 2 == 0);
      }

      bool contains(vector<int> v, int n) {
        for (auto it: v) {
          if(it == n) { return true; };
        }
        return false;
      }

      atomic <int> cnt;

      void back(vector<int> sol, int T, int n) {
        if(sol.size() == n) {
          if(check(sol)) {
            cnt++;
          }
          return;
        }
        if(T == 1) {
          for(int i=1; i<=n; ++i) {
            if(contains(sol, i)) continue;
            sol.push_back(i);
            back(sol, T, n);
            sol.pop_back();
          }
        } else {
          vector<int> x(sol);
          thread t([&]() {
            for(int i=1; i<=n; i+=2) {
              if(contains(x, i)) continue;
              x.push_back(i);
              back(x, T/2, n);
              x.pop_back();
            }
          });
          for(int i=2; i<=n; i += 2) {
            if(contains(sol, i)) continue;
            x.push_back(i);
            back(sol, T / 2, n);
            sol.pop_back();
          }
          t.join();
        }
      }

      int main() {
        back(vector<int>(), 2, 3);
        cout << cnt.load();
      }
      `,
      read: false
    },
    {
      id: 7,
      name: 'Feb 2018 Subject No.1',
      index: 'Problem 2',
      question:
      `
      Consider the following code for a thread pool. Find the concurrency issue and fix it. Also, add a mechanism to end the threads at shutdown.
      `,
      answer:
      `
      class ThreadPool {
        condition_variable cv;
        mutex mtx;
        list<function<void()>> work;
        vector<thread> threads;
      
        void run() {
          unique_lock<mutex> lck(mtx);
          while(true) {
            if(work.empty()) {
             cv.wait(lck);
           } else {
             function<void()> vi = work.front();
             work.pop_front();
             vi();
           }
         }
       }
      
      public:
        explicit ThreadPool(int n) {
          threads.resize(n);
          for(int i=0; i<n; ++i) {
            threads.emplace_back([this]() { run(); });
          }
        }
        void enqueue(function<void()> f) {
          unique_lock<mutex> lck(mtx);
          work.push_back(f);
          cv.notify_one();
        }
      };

      Answer:

      class ThreadPool {
        condition_variable cv;
        mutex mtx;
        list<function<void()>> work;
        vector<thread> threads;
        bool done = false;
      
        void run() {
          unique_lock<mutex> lck(mtx);
          while(!done) {
            if(work.empty()) {
              cv.wait(lck);
            } else {
              function<void()> vi = work.front();
              work.pop_front();
              lck.unlock();
              vi();
              lck.lock();
            }
          }
        }
      
      public:
        explicit ThreadPool(int n) {
          threads.resize(n);
          for(int i=0; i<n; ++i) {
            threads.emplace_back([this]() { run(); });
          }
        }
        ~ThreadPool() {
          unique_lock<mutex> lck(mtx);
          done = true;
          cv.notify_all();
          lck.unlock();
          for(thread& t : threads) {
            t.join();
          }
        }
        void enqueue(function<void()> f) {
          unique_lock<mutex> lck(mtx);
          work.push_back(f);
          cv.notify_one();
        }
      };
      - Added a variable done to signal the threads to stop running when the thread pool is destroyed.
      - Used !done instead of true in the while loop to stop the threads when the thread pool is destroyed.
      - Released the lock in run before calling the task to avoid deadlocks.
      - Added a destructor to properly shut down the threads.
      `,
      read: false
    },
    {
      id: 8,
      name: 'Feb 2018 Subject No.1',
      index: 'Problem 3',
      question: 
      `
      Write a parallel or distributed program that finds all the prime numbers up to N. Hint: serially produce all the prime numbers up to sqrt(N), and distribute
      them to all threads or processes.
      `,
      answer: 
      `
      public class PrimeFinderThreads {
        private static final int N = 10000;
        private static final int sqrtN = (int) Math.sqrt(N);
        private static final AtomicInteger count = new AtomicInteger();
    
        private static final int numThreads = 24;
    
        public void startThreads() {
            Thread[] threads = new Thread[numThreads];
            int chunk = N / numThreads;
    
            for (int i = 0; i < numThreads; i++) {
    
                int start = i * chunk;
                int end;
                if (i == numThreads - 1) {
                    end = N;
                } else {
                    end = (i + 1) * chunk - 1;
                }
                System.out.println("start = " + start + "; end = " + end);
                threads[i] = new Thread(() -> {
                    for (int j = start; j <= end; j++) {
                        if (isPrime(j)) {
                            count.incrementAndGet();
                        }
                    }
                });
                threads[i].start();
            }
    
            for (int i = 0; i < numThreads; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    System.out.println("Thread interrupted: " + e.getMessage());
                }
            }
    
            System.out.println("Number of primes up to sqrt(" + N + "): " + count.get());
        }
    
          private static boolean isPrime(int n) {
              if (n <= 1) return false;
              if (n == 2) return true;
              if (n % 2 == 0) return false;
              int k = 3;
              while (k * k <= n && n % k != 0) {
                  k += 2;
              }
              return k * k > n;
          }
      }
      `,
      read: false
    },
    {
      id: 9,
      name: 'Feb 2018 Subject No.2',
      index: 'Problem 1',
      question: 
      `
      Write a parallel (distributed or local, at your choice) program that computes the (discrete) convolution of a vector with another vector. 
      The convolution is defined as ri=(sum from j=0 to N-1)aj bi-j. All three vectors are of length N and, for simplicity, i - j shall be taken modulo N.
      `,
      answer: 
      `
      Here is an example implementation in C++ using MPI:

      #include <mpi.h>
      #include <vector>
      #include <cmath>
      #include <iostream>

      const int N = 100; // Length of vectors

      // Compute the convolution of two vectors using MPI
      std::vector<double> convolution(const std::vector<double> &a, const std::vector<double> &b) {
        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int chunk_size = ceil(N / size);
        int start = rank * chunk_size;
        int end = start + chunk_size;
        end = std::min(end, N);

        std::vector<double> result(N);
        for (int i = start; i < end; i++) {
          result[i] = 0.0;
          for (int j = 0; j < N; j++) {
          result[i] += a[j] * b[(i - j + N) % N];
        }
      }

        std::vector<double> final_result(N);
        MPI_Allreduce(result.data(), final_result.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return final_result;
      }

      int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        std::vector<double> a(N);
        std::vector<double> b(N);

        // Fill the vectors a and b with values
        // ...

        auto result = convolution(a, b);

        if (rank == 0) {
          std::cout << "Result of convolution: ";
          for (auto r : result) {
            std::cout << r << " ";
          }
          std::cout << std::endl;
        }

        MPI_Finalize();
        return 0;
      }
      `,
      read: false
    },
    {
      id: 10,
      name: 'Feb 2018 Subject No.2',
      index: 'Problem 2',
      question:
      `
      Consider the following code for enqueueing a continuation on a future. Identify and fix the thread-safety issue.
      `,
      answer: 
      `
      template<typename T> class Future {
        list<function<void (T)>> continuations;
        T val;
        bool hasValue;
      public:
        Future() hasValue(false) {}
        void set (T v) {
          val = V;
          hasValue = true;
          for (function<void (T)>& f: continuations) {
            f(v);
          }
          continuations.clear();
        }
      
        void addContinuation (function<void (T)> f) { 
          if (hasValue) {
            f(val);
          } else {
            continuations.push_back(f);
          }
        }
      };

      Answer:

      The thread-safety issue in the code is that the set method and the addContinuation method are not protected by a synchronization mechanism. If one thread is setting the value of the future while another thread is enqueueing a continuation, it's possible for the continuation to be executed before the value has been set, which could result in incorrect behavior.
      To fix this, we need to ensure that access to the val variable, hasValue flag, and the continuations list is synchronized across multiple threads. We can use a mutex to ensure that access to these shared resources is mutually exclusive.

      template<typename T> class Future {
        list<function<void (T)>> continuations;
        T val;
        bool hasValue;
        mutex mtx;
        
      public:
        Future() hasValue(false) {}
        void set (T v) {
          unique_lock<mutex> lck(mtx);
          val = V;
          hasValue = true;
          for (function<void (T)>& f: continuations) {
            f(v);
          }
          continuations.clear();
        }
        
        void addContinuation (function<void (T)> f) {
          unique_lock<mutex> lck(mtx);
          if (hasValue) {
            f(val);
          } else {
            continuations.push_back(f);
          }
          }
        };
      `,
      read: false
    },
    {
      id: 11,
      name: 'Feb 2018 Subject No.2',
      index: 'Problem 3',
      question: `Write a parallel algorithm that computes the product of 2 matrices.`,
      answer: 
      `
      public class MatrixMultiplication {

        private static final int N = 1000; // matrix dimension
    
        public final int[][] A = new int[N][N];
        public final int[][] B = new int[N][N];
        private static final int[][] result = new int[N][N];
    
        private static final int[][] result2 = new int[N][N];
    
        private static final int NUM_THREADS = 23;
    
        private class MultiplyTask implements Runnable {
            private final int startRow;
            private final int endRow;
    
            public MultiplyTask(int _startRow, int _endRow) {
                startRow = _startRow;
                endRow = _endRow;
            }
    
            @Override
            public void run() {
                for (int i = startRow; i < endRow; i ++) {
                    for (int j = 0; j < N; j ++) {
                        result[i][j] += A[i][j] * B[i][j];
                    }
                }
            }
        }
    
          private void initMatrices() {
              Random random = new Random();
              for (int i = 0; i < N; i ++) {
                  for (int j = 0; j < N; j ++) {
                      A[i][j] = random.nextInt(10);
                      B[i][j] = random.nextInt(10);
                  }
              }
          }
      
          private void computeIterative() {
              for (int i = 0; i < N; i ++) {
                  for (int j = 0; j < N; j ++) {
                      result2[i][j] += A[i][j] * B[i][j];
                  }
              }
          }
      
          private boolean correctnessCheck() {
              for (int i = 0; i < N; i ++) {
                  for (int j = 0; j < N; j ++) {
                      if (result[i][j] != result2[i][j]) {
                          return false;
                      }
                  }
              }
              return true;
          }
      
          public void startThreads() {
              initMatrices();
              Thread[] threads = new Thread[NUM_THREADS];
              int k = N / NUM_THREADS;
      
              for (int i = 0; i < NUM_THREADS; i ++) {
                  int startRow = i * k;
                  int endRow;
                  if (i == NUM_THREADS - 1) {
                      endRow = N;
                  } else {
                      endRow = (i + 1) * k;
                  }
      
                  threads[i] = new Thread(new MultiplyTask(startRow, endRow));
                  threads[i].start();
              }
      
              for (int i = 0; i < NUM_THREADS; i ++) {
                  try {
                      threads[i].join();
                  } catch (InterruptedException e) {
                      throw new RuntimeException(e);
                  }
              }
      
              var s = new StringBuilder();
              for (int i = 0; i < N; i ++) {
                  for (int j = 0; j < N; j ++) {
                      s.append(result[i][j]).append(" ");
                  }
                  s.append("newline");
              }
              System.out.println(s);
      
              computeIterative();
              var flag = correctnessCheck();
              System.out.println(flag);
          }
      }
      `,
      read: false
    },
    {
      id: 12,
      name: 'Feb 2018 Subject No.3',
      index: 'Problem 1',
      question: 
      `
      Write a parallel or distributed program that counts the number of permutations of N that satisfy a given property. 
      You have a function bool pred (vector<int> const& v) that verifies if a given permutation satisfies the property. 
      Your program shall call that function once for each permutation and count the number of times it returns true. 
      `,
      answer: 
      `
      #include <mpi.h>
      #include <iostream>
      #include <vector>
      #include <algorithm>

      using namespace std;

      const int root = 0;
      bool pred(vector<int> const& v);

      int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int N;
        if (rank == root) {
          cin >> N;
        }
        MPI_Bcast(&N, 1, MPI_INT, root, MPI_COMM_WORLD);

        vector<int> perm(N);
        for (int i = 0; i < N; i++) {
          perm[i] = i;
        }

        int count = 0;
        int chunk_size = N / world_size;
        int start = chunk_size * rank;
        int end = start + chunk_size;
        if (rank == world_size - 1) {
          end = N;
        }

        do {
          for (int i = start; i < end; i++) {
            vector<int> subperm(perm.begin(), perm.begin() + i + 1);
            if (pred(subperm)) {
              count++;
            }
          }
        } while (next_permutation(perm.begin(), perm.end()));

        int total_count = 0;
        MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        if (rank == root) {
          cout << total_count << endl;
        }

        MPI_Finalize();
        return 0;
      }

      bool pred(vector<int> const& v) {
        // Your code to check if the permutation satisfies the property
        return true;
      } 
      `,
      read: false
    },
    {
      id: 13,
      name: 'Feb 2018 Subject No.3',
      index: 'Problem 2',
      question: 'Consider the following code for enqueueing a work item to a thread pool. Find the concurrency issue and fix it. Also, add a mechanism to end the threads at shutdown.',
      answer: 
      `
      class ThreadPool {
        condition_variable cv;
        mutex mtx;
        list<function<void()>> work; 
        vector<thread> threads;
        void run() {
          while(true) {
            if (work.empty()) {
              unique_lock<mutex> lck(mtx); 
              cv.wait(lck);
            } else {
              function<void()> vi = work.front(); 
              work.pop_front();
              wi();
            }
          }
        }
      public:
        explicit ThreadPool (int n) {
          threads.resize(n);
          for(int i=0; i<n; ++i) {
            threads.emplace_back([this] () {run();});
          }
        }
        void enqueue (function<void()> f) {
          unique_lock<mutex> lck (mtx):
          work.push_back(f);
          cv.notify_one();
        }
      }

      Answer:

      The code is trying to create a thread pool, where a user can enqueue work items that will be executed by the pool threads.

      Concurrency issue:

      The code does not have any mechanism to end the threads at shutdown, which could result in memory leaks or resource starvation.
      Fix:

      To fix the issue, you can add a flag variable named "done" that is used to indicate the threads to shut down.
      At shutdown, the flag can be set to true and broadcasted to all threads using cv.notify_all().
      In the run function, you can check the value of the done flag and exit the loop when it is true.

      class ThreadPool {
        condition_variable cv;
        mutex mtx;
        list<function<void()>> work; 
        vector<thread> threads;
        bool done = false;
        void run() {
          while(true) {
            if (work.empty() && !done) {
              unique_lock<mutex> lck(mtx); 
              cv.wait(lck);
            } else if (done) {
              break;
            } else {
              function<void()> vi = work.front(); 
              work.pop_front();
              wi();
            }
          }
        }
      public:
        explicit ThreadPool (int n) {
          threads.resize(n);
          for(int i=0; i<n; ++i) {
            threads.emplace_back([this] () {run();});
          }
        }
        void enqueue (function<void()> f) {
          unique_lock<mutex> lck (mtx):
          work.push_back(f);
          cv.notify_one();
        }
        void shutdown() {
          {
            unique_lock<mutex> lck (mtx);
            done = true;
          }
          cv.notify_all();
          for (auto& t : threads) {
            t.join();
          }
        }
      }
      `,
      read: false
    },
    {
      id: 14,
      name: 'Feb 2018 Subject No.3',
      index: 'Problem 3',
      question: 'Write a parallel algorithm that computes the product of two big numbers.',
      answer: 
      `
      import java.math.BigInteger;
      import java.util.concurrent.Callable;
      import java.util.concurrent.ExecutorService;
      import java.util.concurrent.Executors;
      import java.util.concurrent.Future;

      public class BigNumberProduct {
          static class Multiplier implements Callable<BigInteger> {
              BigInteger num1;
              BigInteger num2;

              Multiplier(BigInteger num1, BigInteger num2) {
                  this.num1 = num1;
                  this.num2 = num2;
              }

              @Override
              public BigInteger call() {
                  return num1.multiply(num2);
              }
          }

          public static void main(String[] args) throws Exception {
              BigInteger num1 = new BigInteger("1234567890123456789012345678901234567890");
              BigInteger num2 = new BigInteger("9876543210987654321098765432109876543210");
              int numThreads = 4;
              ExecutorService executor = Executors.newFixedThreadPool(numThreads);
              Future<BigInteger>[] futures = new Future[numThreads];
              for (int i = 0; i < numThreads; i++) {
                  BigInteger subNum1 = num1.divide(BigInteger.valueOf(numThreads)).multiply(BigInteger.valueOf(i));
                  BigInteger subNum2 = num2.divide(BigInteger.valueOf(numThreads)).multiply(BigInteger.valueOf(i));
                  futures[i] = executor.submit(new Multiplier(subNum1, subNum2));
              }
              BigInteger product = BigInteger.ZERO;
              for (int i = 0; i < numThreads; i++) {
                  product = product.add(futures[i].get());
              }
              executor.shutdown();
              System.out.println("Product: " + product);
          }
      }
      `,
      read: false

    }
  ];

  private subjectsFrom2017: Exercise[] = [this.exercises[0], this.exercises[1], this.exercises[2], this.exercises[3], this.exercises[4], this.exercises[5]];
  private subjectsFrom2018: Exercise[] = [this.exercises[6], this.exercises[7], this.exercises[8], this.exercises[9], this.exercises[10], this.exercises[11],
                                          this.exercises[12], this.exercises[13], this.exercises[14]];

  constructor() { }

  public getExercisesByYear(year: Number): Exercise[] {
    switch (year) {
      case 2017:
        return this.subjectsFrom2017;
      case 2018:
        return this.subjectsFrom2018;
      default:
        return this.exercises;
    }
  }

  public getExerciseById(id: number): Exercise {
    return this.exercises[id];
  }
}
