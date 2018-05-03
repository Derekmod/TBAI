import sys

class PriorityQueue(object):
    def __init__(self, hash_function, value_function, low_best=True):
        self._items = []
        self._map = dict()
        self._size = 0

        self._first_empty = 0
        self._last_filled = -1

        self._hash_function = hash_function
        self._value_function = value_function
        if not low_best:
            self._value_function = lambda item: -value_function(item)

    def pop(self):
        self._size -= 1
        ret = self._items[0]
        if self._last_filled == 0:
            self._items = []
            self._map = dict()

            self._first_empty = 0
            self._last_filled = -1
            return ret

        self._swap(0, self._last_filled)
        self._items[self._last_filled] = None
        self._first_empty = min(self._first_empty, self._last_filled)
        del self._map[self._hash_function(ret)]

        self._sink(0)

        while self._items[self._last_filled] is None and self._last_filled >= 0:
            self._last_filled -= 1

        return ret

    def add(self, item):
        self._size += 1
        idx = self._first_empty
        if idx >= len(self._items):
            idx = len(self._items)
            self._items.append(None)
        self._items[idx] = item
        self._map[self._hash_function(item)] = idx
        if idx > self._last_filled:
            self._last_filled = idx
        
        self._bubble(idx)

        while self._first_empty < len(self._items) and self._items[self._first_empty] is not None:
            self._first_empty += 1
        if self._first_empty >= len(self._items):
            self._first_empty = len(self._items)
            self._items.append(None)

    def addNoDuplicate(self, item):
        hash = self._hash_function(item)
        if hash in self._map:
            self._items[self._map[hash]] = item
            self.update(hash)
        else:
            self.add(item)

    def peek(self):
        return self._items[0]

    def update(self, hash):
        idx = self._map[hash]
        idx = self._bubble(idx)
        idx = self._sink(idx)

    def get(self, hash):
        return self._items[self._map[hash]]

    @property
    def items(self):
        return iter(self._items)

    def _sink(self, idx):
        val = self._value_function(self._items[idx])
        while True:
            child_indices = [2*idx + 1, 2*idx + 2]
            swapped = False
            for child_idx in child_indices:
                if child_idx >= len(self._items):
                    continue
                child_item = self._items[child_idx]
                if child_item is None:
                    continue
                child_val = self._value_function(child_item)
                if val <= child_val:
                    continue
                self._swap(idx, child_idx)
                idx = child_idx
                swapped = True
                break
            if not swapped:
                break
        return idx

    def _bubble(self, idx):
        val = self._value_function(self._items[idx])
        while idx > 0:
            parent_idx = int((idx-1)/2)
            parent_val = self._value_function(self._items[parent_idx])
            if parent_val > val:
                self._swap(idx, parent_idx)
                idx = parent_idx
            else:
                break
        return idx

    def _swap(self, idx1, idx2):
        t_item = self._items[idx2]
        self._items[idx2] = self._items[idx1]
        self._items[idx1] = t_item
                
        self._map[self._hash_function(self._items[idx1])] = idx1
        self._map[self._hash_function(self._items[idx2])] = idx2
        #if not self.validate():
        #    sys.exit()

    def validate(self):
        if self._last_filled >= len(self._items):
            print('"last filled" too high: %d' % self._last_filled)
            return False

        if self._first_empty < 0:
            print('"first empty" too low: %d' % self._first_empty)
            return False

        op = range(len(self._items))
        op.reverse()
        for i in op[:-1]:
            if self._items[i] and not self._items[(i-1)/2]:
                print('slot %d empty with real child: %d' % ((i-1)/2, i))
                return False

        return True

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def next(self):
        if self._size:
            return self.pop()
        else:
            raise StopIteration