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
