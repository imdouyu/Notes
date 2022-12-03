## Leetcode Hot 100
#### [1. 两数之和](https://leetcode.cn/problems/two-sum/)
```cpp
class Solution {
public:
  vector<int> twoSum(vector<int> &nums, int target) {
    unordered_map<int, int> map_;
    for (int i = 0; i < nums.size(); i++) {
      if (map_.find(target - nums[i]) != map_.end()) {
        return {i, map_[target - nums[i]]};
      }
      map_[nums[i]] = i;
    }
    return {};
  }
};
```
哈希表，一次遍历
#### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)
```cpp
class Solution {
public:
  ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    ListNode *head = new ListNode(-1);
    ListNode *l3 = head;
    int carry = 0;
    while (l1 || l2) {
      int n1 = l1 ? l1->val : 0;
      int n2 = l2 ? l2->val : 0;
      ListNode *node = new ListNode((n1 + n2 + carry) % 10);
      carry = (n1 + n2 + carry) / 10;
      l3->next = node;
      l3 = l3->next;
      if (l1) {
        l1 = l1->next;
      }
      if (l2) {
        l2 = l2->next;
      }
    }
    if (carry) {
      l3->next = new ListNode(1);
      l3 = l3->next;
    }
    return head->next;
  }
};

```
模拟+一次遍历      
链表题目往往会需要一个dummy头节点,减少corner case    
#### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)
```cpp
class Solution {
public:
  int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> map;
    int res = 0;
    for (int i = 0, j = 0; i < s.size(); i++) {
      if (map.find(s[i]) != map.end()) {
        j = max(j, map[s[i]] + 1);
      }
      map[s[i]] = i;
      res = max(res, i - j + 1);
    }
    return res;
  }
};

```
哈希表+滑动窗口
#### [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)
方法一：对顶堆，时间复杂度不符合要求  
方法二：二分查找，对于要求$O(logn)$时间复杂度的题目，一般往往需要二分查找
#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
方法一：dp       
```cpp
class Solution {
public:
  string longestPalindrome(string s) {
    vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
    int start = 0, maxlen = 0;
    for (int j = 0; j < s.size(); j++) {
      for (int i = 0; i <= j; i++) {
        if (i == j) {
          dp[i][j] = 1;
        } else if (j - i + 1 == 2) {
          dp[i][j] = s[i] == s[j];
        } else {
          dp[i][j] = s[i] == s[j] && dp[i + 1][j - 1];
        }
        if (dp[i][j]) {
          start = maxlen < j - i + 1 ? i : start;
          maxlen = max(maxlen, j - i + 1);
        }
      }
    }
    return s.substr(start, maxlen);
  }
};

```
方法二：中心扩展法       
```cpp
class Solution {
public:
  string longestPalindrome(string s) {
    int begin, max_len = 0;
    int len = s.size();
    for (int i = 0; i < len; i++) {
      // 奇数
      int l = i - 1, r = i + 1;
      while (l >= 0 && r < len && s[l] == s[r]) {
        l--;
        r++;
      }
      if (max_len < r - l - 1) {
        begin = l + 1;
        max_len = r - l - 1;
      }
      // 偶数
      l = i;
      r = i + 1;
      while (l >= 0 && r < len && s[l] == s[r]) {
        l--;
        r++;
      }
      if (max_len < r - l - 1) {
        begin = l + 1;
        max_len = r - l - 1;
      }
    }
    return s.substr(begin, max_len);
  }
};
```
#### [10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)

#### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)
```cpp
class Solution {
public:
  int maxArea(vector<int> &height) {
    long long res = INT_MIN;
    for (int i = 0, j = height.size() - 1; i < j;) {
      res = max(res, (long long)min(height[i], height[j]) * (j - i));
      // 宽度减小的情况下，尽可能增大高度，所以移动短板(贪心)
      if (height[i] < height[j]) {
        i++;
      } else {
        j--;
      }
    }
    return res;
  }
};

```
双指针+贪心，短板限制储水的最大容量
#### [15. 三数之和](https://leetcode.cn/problems/3sum/)
```cpp
class Solution {
public:
  vector<vector<int>> threeSum(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    for (int i = 0; i < nums.size(); i++) {
      if (nums[i] > 0) {
        break;
      }
      if (i > 0 && nums[i] == nums[i - 1]) {
        continue;
      }
      for (int j = i + 1, k = nums.size() - 1; j < k;) {
        int sum = nums[i] + nums[j] + nums[k];
        if (sum == 0) {
          res.push_back({nums[i], nums[j], nums[k]});
          while (j < k && nums[j] == nums[j + 1]) {
            j++;
          }
          while (j < k && nums[k] == nums[k - 1]) {
            k--;
          }
          j++;
          k--;
        } else if (sum < 0) {
          j++;
        } else {
          k--;
        }
      }
    }
    return res;
  }
};
```
排序+双指针，左右指针都得去重
#### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)
```cpp
class Solution {
  vector<string> res;
  string str;
  unordered_map<char, vector<char>> hash = {
      {'2', {'a', 'b', 'c'}}, {'3', {'d', 'e', 'f'}},
      {'4', {'g', 'h', 'i'}}, {'5', {'j', 'k', 'l'}},
      {'6', {'m', 'n', 'o'}}, {'7', {'p', 'q', 'r', 's'}},
      {'8', {'t', 'u', 'v'}}, {'9', {'w', 'x', 'y', 'z'}}};

public:
  vector<string> letterCombinations(string digits) {
    if (digits.size() == 0) {
      return res;
    }
    bt(digits, 0);
    return res;
  }

  void bt(string digits, int start) {
    if (str.length() == digits.length()) {
      res.push_back(str);
      return;
    }
    auto v = hash[digits[start]];
    for (int j = 0; j < v.size(); j++) {
      str.push_back(v[j]);
      bt(digits, start + 1);
      str.pop_back();
    }
  }
};
```
回溯法
#### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)
```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *fast = dummy, *slow = dummy;
        int i = 0;
        while(fast->next && i < n) {
            fast = fast->next;
            i++;
        }
        while(fast->next) {
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        return dummy->next;
    }
};
```
核心思想是通过遍历找到要移除节点的前一个节点curr，使其curr->next指向curr->next->next  
快慢指针+dummy head
#### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)
```cpp
class Solution {
public:
  bool isValid(string s) {
    stack<char> stk;
    unordered_map<char, char> m = {{'(', ')'}, {'{', '}'}, {'[', ']'}};
    for (const auto &ch : s) {
      if (m.find(ch) != m.end()) {
        stk.push(m[ch]);
      } else {
        if (stk.empty() || stk.top() != ch) {
          return false;
        }
        stk.pop();
      }
    }
    return stk.empty();
  }
};
```
#### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)
```cpp
class Solution {
public:
  ListNode *mergeTwoLists(ListNode *list1, ListNode *list2) {
    // dummy head
    ListNode *head = new ListNode(0);
    ListNode *prev = head;
    while (list1 && list2) {
      if (list1->val < list2->val) {
        prev->next = list1;
        list1 = list1->next;
      } else {
        prev->next = list2;
        list2 = list2->next;
      }
      prev = prev->next;
    }
    if (list1) {
      prev->next = list1;
    }
    if (list2) {
      prev->next = list2;
    }
    return head->next;
  }
};
```
#### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)
```cpp
class Solution {
  string str;
  vector<string> res;

public:
  vector<string> generateParenthesis(int n) {
    bt(str, n, 0, 0);
    return res;
  }
  void bt(string str, int n, int open, int close) {
    if (open == n && close == n) {
      res.push_back(str);
      return;
    }
    if (open < n) {
      bt(str + '(', n, open + 1, close);
    }
    if (close < open) {
      bt(str + ')', n, open, close + 1);
    }
  }
};
```
回溯法
#### [23. 合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)
```cpp
//  1. 
// struct Comparator
// {
//   bool operator()(const Record& lhs, const Record& rhs)
//   {
//     return lhs.count>rhs.count;
//   }
// };
// 1.
// class Compare{
// public:
//     bool operator() (ListNode * lhs, ListNode * rhs)
//     {
//         return lhs->val>rhs->val;
//     }
//     // https://en.cppreference.com/w/cpp/utility/functional/greater
// };
class Solution {
public:
    // 2.
    // static bool compare(ListNode * lhs, ListNode * rhs) {
    //     return lhs->val > rhs->val;
    // }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *dummy = new ListNode(-1);
        ListNode *curr = dummy;
        // priority_queue<ListNode *, vector<ListNode *>, Compare> pq;
        
        
        // priority_queue<ListNode *, vector<ListNode *>, decltype(&compare)> pq(compare);
        // 3.
        auto compare = [](ListNode * lhs, ListNode * rhs) { 
            return lhs->val > rhs->val;
        };
        priority_queue<ListNode *, vector<ListNode *>, decltype(compare)> pq(compare);
        for(int i = 0; i < lists.size(); i++) {
            if(lists[i] != nullptr) {
                pq.push(lists[i]);
            }
        }
        while(!pq.empty()) {
            ListNode *node = pq.top();
            pq.pop();
            curr->next = node;
            curr = curr->next;
            if(node->next != nullptr) {
                pq.push(node->next);
            }
        }
        return dummy->next;
    }
};
```
优先队列  
cmp的几种写法:  
- struct/class  
- 静态函数，不推荐  
- lambda  
#### [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)
#### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)
```cpp
class Solution {
public:
  int longestValidParentheses(string s) {
    int ans = 0;
    // 栈内存左括号的索引
    stack<int> st;
    for (int i = 0, start = -1; i < s.size(); i++) {
      if (s[i] == '(') {
        st.push(i);
      } else {
        if (!st.empty()) {
          st.pop();
          if (!st.empty()) {
            ans = max(ans, i - st.top());
          } else {
            ans = max(ans, i - start);
          }
        } else {
          // 此位置表示右括号的数量多余左括号的数量
          start = i;
        }
      }
    }
    return ans;
  }
};
```
#### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)
```cpp
class Solution {
public:
  int search(vector<int> &nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      if (nums[mid] == target) {
        return mid;
      }
      if (nums[left] <= nums[mid]) {
        if (target < nums[mid] && target >= nums[left]) {
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      } else {
        if (target > nums[mid] && target <= nums[right]) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return -1;
  }
};
```
二分查找，类似题目81，153，154  
33/81:  
![33/81](images/1.png)   
153/154:   
![153/154](images/2.png)
#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)
```cpp
class Solution {
public:
  vector<int> searchRange(vector<int> &nums, int target) {
    return {lower_bound(nums, target), upper_bound(nums, target)};
  }

  int upper_bound(vector<int> &nums, int target) {
    int low = 0, high = nums.size() - 1;
    while (low < high) {
      // 取上边界，防止死循环
      int mid = low + (high - low + 1) / 2;
      if (nums[mid] <= target) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    // 确保nums[low]不下溢
    return nums.size() > 0 && nums[low] == target ? low : -1;
  }
  int lower_bound(vector<int> &nums, int target) {
    int low = 0, high = nums.size() - 1;
    while (low < high) {
      // 取下边界，防止死循环
      int mid = low + (high - low) / 2;
      if (nums[mid] >= target) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return nums.size() > 0 && nums[low] == target ? low : -1;
  }
};
```
#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)
```cpp
class Solution {
  vector<vector<int>> res;
  vector<int> templist;

public:
  vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
    sort(candidates.begin(), candidates.end());
    bt(candidates, target, 0);
    return res;
  }

  void bt(vector<int> &candidates, int remain, int start) {
    if (remain < 0) {
      return;
    } else if (remain == 0) {
      res.push_back(templist);
    }
    for (int i = start; i < candidates.size(); i++) {
      templist.push_back(candidates[i]);
      bt(candidates, remain - candidates[i], i);
      templist.pop_back();
    }
  }
};
```
回溯法
#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)
```cpp
class Solution {
public:
  int trap(vector<int> &height) {
    int res = 0;
    int leftMax = 0, rightMax = 0;
    int i = 0, j = height.size() - 1;
    while (i < j) {
      leftMax = max(leftMax, height[i]);
      rightMax = max(rightMax, height[j]);
      if (leftMax < rightMax) {
        res += (leftMax - height[i]);
        i++;
      } else {
        res += (rightMax - height[j]);
        j--;
      }
    }
    return res;
  }
};
```
双指针
