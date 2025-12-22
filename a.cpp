/**
 * HPF Graph Scheduling - Ultimate >38M Optimization
 * Strategy: Affinity-Locking + Predictive Load Balancing
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <map>

using namespace std;

// --- Constants ---
const int MAX_NODES = 25;
const int MAX_CORES = 40;
const long long INF_TIME = 4e18;

// --- Data Structures ---
struct Packet {
    int id;
    int type;
    int arrive;
    int timeout;
    long long deadline;
    
    int current_path_idx;
    int last_core;      // 0 for Input
    long long ready_time;
    
    int special_mask;
    bool queried_special;
    bool active;
};

struct NodeConfig {
    int max_batch;
    vector<int> costs; 
    int best_batch;
    double min_avg_cost;
};

struct NodeQueue {
    vector<int> pending; 
    vector<int> ready;   
    bool dirty;          
};

// --- Globals ---
vector<vector<int>> packet_paths(16); 
vector<Packet> packets;
NodeConfig nodes[MAX_NODES];
NodeQueue node_queues[MAX_NODES];
long long core_free_time[MAX_CORES]; 
int special_costs[8];
int mask_cost[256];

int num_cores, c_switch, c_r;

// Scheduler State
long long G_last_issue_time = 1; 
long long G_receive_unlock_time = 0;
int finished_count = 0;
int received_count = 0;
int active_packets = 0;

// --- Helpers ---
long long calculate_exec_cost(int node_id, int core_id, const vector<int>& batch) {
    if (batch.empty()) return 0;
    long long cost = nodes[node_id].costs[batch.size()];
    long long extra = 0;
    for (int pid : batch) {
        if (packets[pid].last_core != core_id) extra += c_switch;
        if (node_id == 8) extra += mask_cost[packets[pid].special_mask];
    }
    return cost + extra;
}

void sync_queues(long long now) {
    for (int i = 1; i <= 20; ++i) {
        if (node_queues[i].pending.empty()) continue;
        vector<int> next_pending;
        next_pending.reserve(node_queues[i].pending.size());
        for (int pid : node_queues[i].pending) {
            if (packets[pid].ready_time <= now) {
                node_queues[i].ready.push_back(pid);
                node_queues[i].dirty = true;
            } else {
                next_pending.push_back(pid);
            }
        }
        node_queues[i].pending = move(next_pending);
    }
}

// --- I/O Wrappers ---
void output_R(long long t) {
    long long actual = max(t, G_last_issue_time);
    cout << "R " << actual << endl;
    G_last_issue_time = actual;
}
void output_Q(long long t, int pid) {
    long long actual = max(t, G_last_issue_time);
    cout << "Q " << actual << " " << pid << endl;
    G_last_issue_time = actual;
}
void output_E(long long t, int core, int node, int size, const vector<int>& batch) {
    long long actual = max(t, G_last_issue_time);
    cout << "E " << actual << " " << core << " " << node << " " << size;
    for(int id : batch) cout << " " << id;
    cout << endl;
    G_last_issue_time = actual;
}
void output_F() { cout << "F" << endl; }

// --- Initialization ---
void load_data() {
    for (int i = 1; i <= 7; ++i) {
        int l; cin >> l;
        packet_paths[i].resize(l);
        for (int j = 0; j < l; ++j) cin >> packet_paths[i][j];
    }
    for (int i = 1; i <= 20; ++i) {
        cin >> nodes[i].max_batch;
        nodes[i].costs.resize(nodes[i].max_batch + 1);
        nodes[i].min_avg_cost = 1e18;
        nodes[i].best_batch = 1;
        for (int j = 1; j <= nodes[i].max_batch; ++j) {
            cin >> nodes[i].costs[j];
            double avg = (double)nodes[i].costs[j] / j;
            if (avg <= nodes[i].min_avg_cost) {
                nodes[i].min_avg_cost = avg;
                nodes[i].best_batch = j;
            }
        }
    }
    for (int i = 0; i < 8; ++i) cin >> special_costs[i];
    for (int m = 0; m < 256; ++m) {
        int sum = 0;
        for (int b = 0; b < 8; ++b) if ((m >> b) & 1) sum += special_costs[b];
        mask_cost[m] = sum;
    }
    cin >> num_cores >> c_switch >> c_r;
}

// --- CORE SELECTION ---
struct Selection {
    int core;
    vector<int> batch;
    long long finish_time;
};

Selection find_best_core_for_node(int node_id, long long current_time) {
    if (node_queues[node_id].ready.empty()) return {-1, {}, INF_TIME};

    if (node_queues[node_id].dirty) {
        sort(node_queues[node_id].ready.begin(), node_queues[node_id].ready.end(), 
             [](int a, int b) { return packets[a].deadline < packets[b].deadline; });
        node_queues[node_id].dirty = false;
    }

    long long head_deadline = packets[node_queues[node_id].ready[0]].deadline;
    long long slack = head_deadline - current_time;
    int q_size = node_queues[node_id].ready.size();

    // System States
    bool panic = (slack < 2000) || (active_packets > num_cores * 18);
    bool heavy = (active_packets > num_cores * 8);
    
    // Batch Sizing
    int limit = nodes[node_id].max_batch;
    if (!panic && !heavy) {
        // Optimal batch size logic
        int optimal = nodes[node_id].best_batch;
        // If queue is much larger than optimal, increase limit to avoid backlog
        if (q_size > optimal * 2) limit = min(nodes[node_id].max_batch, q_size);
        else limit = min(q_size, optimal);
    }
    limit = min(limit, q_size);

    int best_core = -1;
    vector<int> best_batch;
    double best_metric = 1e18;
    long long best_finish_time = INF_TIME; 

    // Iterate Cores
    for (int c = 1; c <= num_cores; ++c) {
        long long core_avail = max(current_time, core_free_time[c]);
        
        // Dynamic Wait Tolerance
        long long wait_tol = 0;
        if (!panic) {
            if (heavy) wait_tol = c_switch; 
            else wait_tol = c_switch * 2; 
        }
        
        if (core_avail > current_time + wait_tol) continue;
        if (best_core != -1 && core_avail >= best_finish_time) continue;

        long long start_t = max(G_last_issue_time, core_avail);

        // --- Batch Construction ---
        vector<int> cand_batch; cand_batch.reserve(limit);
        vector<int> others; others.reserve(limit);
        
        int scan_depth = panic ? limit : min(q_size, limit * 5);
        
        for (int k = 0; k < scan_depth; ++k) {
            int pid = node_queues[node_id].ready[k];
            // High Urgency Bypass
            if (packets[pid].deadline - start_t < 2500) {
                 if ((int)cand_batch.size() < limit) cand_batch.push_back(pid);
            } else {
                 // Affinity Check
                 if (packets[pid].last_core == c) {
                     if ((int)cand_batch.size() < limit) cand_batch.push_back(pid);
                 } else {
                     if ((int)others.size() < limit) others.push_back(pid);
                 }
            }
        }
        
        // Fill from others if allowed
        // In heavy/panic mode, always fill. In normal mode, only if affinity batch is too small.
        bool fill = panic || heavy || (cand_batch.size() < nodes[node_id].best_batch / 2);
        
        if (fill) {
            for (int pid : others) {
                if ((int)cand_batch.size() >= limit) break;
                cand_batch.push_back(pid);
            }
        }
        
        if (cand_batch.empty()) continue;

        long long actual_start = start_t;
        for(int pid : cand_batch) actual_start = max(actual_start, packets[pid].ready_time);
        
        long long cost = calculate_exec_cost(node_id, c, cand_batch);
        long long finish = actual_start + cost;
        
        // --- Metric Calculation ---
        // Base: Finish Time
        double metric = (double)finish;
        
        if (!panic) {
            // Reward low cost (Affinity)
            metric += (double)cost * 0.7; 
            // Penalize waiting
            metric += (double)(actual_start - current_time) * 0.3;
        }

        if (metric < best_metric) {
            best_metric = metric;
            best_finish_time = finish;
            best_core = c;
            best_batch = cand_batch;
        }
    }
    
    return {best_core, best_batch, best_finish_time};
}

// --- Main Solver ---
void solve_subtask() {
    int n;
    if (!(cin >> n)) return;

    packets.clear(); packets.resize(n + 1000); 
    for (int i = 0; i < MAX_NODES; ++i) {
        node_queues[i].pending.clear();
        node_queues[i].ready.clear();
        node_queues[i].dirty = false;
    }
    for (int c = 0; c < MAX_CORES; ++c) core_free_time[c] = 0;
    
    G_last_issue_time = 1;
    G_receive_unlock_time = 0;
    finished_count = 0;
    received_count = 0;
    active_packets = 0;
    long long current_sim_time = 1;

    // Core availability map for O(1) lookup
    vector<bool> is_free(num_cores + 1);

    while (finished_count < n) {
        long long min_core = INF_TIME;
        for (int c = 1; c <= num_cores; ++c) min_core = min(min_core, core_free_time[c]);
        
        long long next_time = min_core;
        if (received_count < n) next_time = min(next_time, max(G_last_issue_time, G_receive_unlock_time));
        
        current_sim_time = max(current_sim_time, next_time);
        current_sim_time = max(current_sim_time, G_last_issue_time);
        
        sync_queues(current_sim_time);

        // Update core status
        for(int c=1; c<=num_cores; ++c) is_free[c] = (core_free_time[c] <= current_sim_time);

        // Action R
        bool need_receive = false;
        if (received_count < n && G_receive_unlock_time <= current_sim_time) {
             if (active_packets < num_cores * 15) need_receive = true;
             else if (min_core <= current_sim_time + 150) need_receive = true;
        }

        if (need_receive) {
            output_R(current_sim_time);
            int p; cin >> p;
            if (p == -1) exit(0);
            G_receive_unlock_time = G_last_issue_time + c_r;
            long long ready_at = G_receive_unlock_time;

            for (int k = 0; k < p; ++k) {
                int id, arr, type, tout;
                cin >> id >> arr >> type >> tout;
                if (id >= (int)packets.size()) packets.resize(id + 1000);
                packets[id] = {id, type, arr, tout, (long long)arr + tout, 0, 0, max((long long)arr, ready_at), 0, false, true};
                
                if (type >= 1 && type < (int)packet_paths.size() && !packet_paths[type].empty()) {
                    int node = packet_paths[type][0];
                    node_queues[node].pending.push_back(id);
                    active_packets++;
                } else {
                    packets[id].active = false;
                    finished_count++;
                }
            }
            received_count += p;
            continue;
        }

        // Action Q
        bool query_done = false;
        if (!node_queues[8].ready.empty()) {
            if (node_queues[8].dirty) {
                sort(node_queues[8].ready.begin(), node_queues[8].ready.end(), 
                     [](int a, int b) { return packets[a].deadline < packets[b].deadline; });
                node_queues[8].dirty = false;
            }
            for (int pid : node_queues[8].ready) {
                if (!packets[pid].queried_special) {
                    long long tQ = max(G_last_issue_time, packets[pid].ready_time);
                    output_Q(tQ, pid);
                    int mask; cin >> mask;
                    if (mask == -1) exit(0);
                    packets[pid].special_mask = mask;
                    packets[pid].queried_special = true;
                    query_done = true;
                    break;
                }
            }
        }
        if (query_done) continue;

        // Action E: Node Scoring
        int best_node = -1;
        double best_node_score = -1e18;

        for (int i = 1; i <= 20; ++i) {
            if (node_queues[i].ready.empty()) continue;
            if (i == 8 && !packets[node_queues[i].ready[0]].queried_special) continue;

            if (node_queues[i].dirty) {
                sort(node_queues[i].ready.begin(), node_queues[i].ready.end(), 
                     [](int a, int b) { return packets[a].deadline < packets[b].deadline; });
                node_queues[i].dirty = false;
            }
            
            long long slack = packets[node_queues[i].ready[0]].deadline - current_sim_time;
            
            // 1. Urgency
            double urgency = 0;
            if (slack < 1000) urgency = 1e15; 
            else if (slack < 4000) urgency = 1e9 * exp(-slack / 2000.0);
            else urgency = 1e6 * exp(-slack / 6000.0);
            
            // 2. Efficiency
            int qs = node_queues[i].ready.size();
            double eff = (double)qs / nodes[i].best_batch * 4000.0;
            
            // 3. Affinity Bonus (Check top packets)
            double affinity_bonus = 0;
            int check_k = min(qs, 10);
            for(int k=0; k<check_k; ++k) {
                int lc = packets[node_queues[i].ready[k]].last_core;
                if(lc > 0 && is_free[lc]) affinity_bonus += 2000.0;
            }

            // Throttling
            if ((double)qs < nodes[i].best_batch * 0.7 && slack > 6000 && affinity_bonus < 500) eff -= 50000.0;

            double total_score = urgency + eff + affinity_bonus;

            if (total_score > best_node_score) {
                best_node_score = total_score;
                best_node = i;
            }
        }

        if (best_node != -1) {
            Selection sel = find_best_core_for_node(best_node, current_sim_time);
            
            if (sel.core != -1) {
                long long final_start = max(G_last_issue_time, core_free_time[sel.core]);
                for(int pid : sel.batch) final_start = max(final_start, packets[pid].ready_time);
                
                output_E(final_start, sel.core, best_node, sel.batch.size(), sel.batch);
                
                long long dur = calculate_exec_cost(best_node, sel.core, sel.batch);
                long long final_finish = final_start + dur;
                core_free_time[sel.core] = final_finish;
                
                set<int> processed; for(int pid : sel.batch) processed.insert(pid);
                vector<int> rem; rem.reserve(node_queues[best_node].ready.size());
                for(int pid : node_queues[best_node].ready) if(!processed.count(pid)) rem.push_back(pid);
                node_queues[best_node].ready = move(rem);
                
                for(int pid : sel.batch) {
                    packets[pid].last_core = sel.core;
                    packets[pid].ready_time = final_finish;
                    packets[pid].current_path_idx++;
                    if (packets[pid].current_path_idx < (int)packet_paths[packets[pid].type].size()) {
                        int next = packet_paths[packets[pid].type][packets[pid].current_path_idx];
                        node_queues[next].pending.push_back(pid);
                    } else {
                        packets[pid].active = false;
                        finished_count++;
                        active_packets--;
                    }
                }
                continue;
            }
        }

        // Jump Time
        long long jump = INF_TIME;
        for(int c=1; c<=num_cores; ++c) if(core_free_time[c] > current_sim_time) jump = min(jump, core_free_time[c]);
        for(int i=1; i<=20; ++i) {
             if(!node_queues[i].pending.empty()) {
                 for(size_t k=0; k<min((size_t)5, node_queues[i].pending.size()); ++k)
                    if(packets[node_queues[i].pending[k]].ready_time > current_sim_time)
                        jump = min(jump, packets[node_queues[i].pending[k]].ready_time);
             }
        }
        if (received_count < n && G_receive_unlock_time > current_sim_time) jump = min(jump, G_receive_unlock_time);

        if (jump != INF_TIME && jump > current_sim_time) {
            current_sim_time = jump;
        } else {
            current_sim_time++;
        }
        G_last_issue_time = max(G_last_issue_time, current_sim_time);
    }
    output_F();
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    load_data();
    for (int i = 0; i < 5; ++i) solve_subtask();
    return 0;
}