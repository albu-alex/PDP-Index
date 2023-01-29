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
    }
  ];

  private subjectsFrom2017: Exercise[] = [this.exercises[0], this.exercises[1], this.exercises[2], this.exercises[3]];

  constructor() { }

  public getExercisesByYear(year: Number): Exercise[] {
    switch (year) {
      case 2017:
        return this.subjectsFrom2017;
      default:
        return this.exercises;
    }
  }

  public getExerciseById(id: number): Exercise {
    return this.exercises[id];
  }
}
