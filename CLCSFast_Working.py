import sys
import numpy as np

lcs_arr = np.zeros((4096, 2048), dtype=int)
dp_arr = np.zeros((4096, 2048), dtype=int)
paths = np.empty(2048, dtype=object)
max_clcs_length = 0

LEFT = 0
UP = 1
DIAG = 2

class Path():
    def __init__(self):
        # there are situations where a row might have
        # different boundaries depending on if it's
        # a lower bound or upper bound, like when
        # a path has multiple points on one row.
        # lower bounds are a right bound. Take the max.
        # upper bounds are a left bound. Take the min.
        self.l_bounds = np.full(2048, -np.inf)
        self.u_bounds = np.full(2048, np.inf)
        # we can find the CLCS_length associated with a path
        # by storing the number in the bottom and right-most
        # entry of lcs_arr for its window
        self.CLCS_length = 0

# NOTE: l (ex. 0) is the right bound and
# u (ex. m)  is the left bound
def find_shortest_paths(A,B,p,l,u):
#    print "l: ", l
#    print "u: ", u
    if (u-l) <= 1:
        return
    mid = (l+u)/2
    p[mid] = single_shortest_path(A,B,mid,p[l],p[u])
#    print "mid: ", mid
#    print p[mid].l_bounds[mid:mid+len(A)+1]
#    print p[mid].u_bounds[mid:mid+len(A)+1]
    find_shortest_paths(A,B,p,l,mid)
    find_shortest_paths(A,B,p,mid,u)

def compute_p0_pm(A,B):
    # find p0 and pm
    global lcs_arr
    global paths
    
    m = len(A)
    n = len(B)
    A = A + A

    # run LCS
    for i in range(1, 2*m+1):
        for j in range(1,n+1):
            if A[i-1] == B[j-1]:
                lcs_arr[i][j] = lcs_arr[i-1][j-1]+1
                dp_arr[i][j] = DIAG
            elif lcs_arr[i-1][j] > lcs_arr[i][j-1]:
                # up
                lcs_arr[i][j] = lcs_arr[i-1][j]
                dp_arr[i][j] = UP
	    else:
		lcs_arr[i][j] = lcs_arr[i][j-1]
                # left
                dp_arr[i][j] = LEFT
#        if i == 2:
#            print_matrix(lcs_arr, 2*m+1, n+1)
#            sys.exit()
                
    p0 = retrace(0,m,n)
#    print_matrix(lcs_arr,2*m+1,n+1)
#    sys.exit()

    # we need to make pm a new path
    # or any changes we make to p0
    # will be made to pm
    pm = Path()
    pm.start = m
    pm.end = 2*m    
    pm.CLCS_length = p0.CLCS_length
    
    # shift pm bounds down
    pm.l_bounds[m:2*m+1] = p0.l_bounds[0:m+1]
    pm.u_bounds[m:2*m+1] = p0.u_bounds[0:m+1]
    paths[0] = p0
    paths[m] = pm
#    print p0.l_bounds[0:m+1]
#    print p0.u_bounds[0:m+1]
#    print pm.l_bounds[m:2*m+1]
#    print pm.u_bounds[m:2*m+1]
#    sys.exit()
#    print "BOUNDS"
#    for i in range(0,m+1):
#        print p0.bounds[i]

# this runs only once when finding p0
def retrace(start_row, end_row, end_col):
    # starting at the last node of the array, trace back
    # a shortest path
    global max_clcs_length
    curr_path = Path()
    curr_row = end_row
    curr_col = end_col

#    for i in range(start_row + 1, end_row + 1):
#        dp_arr[i][0] = UP
#    for j in range(1, end_col + 1):
#        dp_arr[start_row][i] = LEFT
#    print_matrix(lcs_arr,2*4+1,5+1)    
        
    # initialize paths as a set of bounds
    while True:
        # the min only really matters if we take a left
        curr_path.l_bounds[curr_row] = max(curr_path.l_bounds[curr_row],
                                                curr_col)
        curr_path.u_bounds[curr_row] = min(curr_path.u_bounds[curr_row],
                                                curr_col)

        # start from the rightmost and bottom node
        # and work to the start_row and 0th column
        # to retrace a path
        # the rows will start at different places (0-m)
        # but the ending column will always be 0
        if curr_row == start_row and curr_col == 0:
            break

        if curr_row == start_row:
            curr_col = curr_col-1
        elif curr_col == 0:
            curr_row = curr_row-1
        elif dp_arr[curr_row][curr_col] == UP:
            curr_row = curr_row-1
        elif dp_arr[curr_row][curr_col] == DIAG:
            curr_row = curr_row-1
            curr_col = curr_col-1
            curr_path.CLCS_length = curr_path.CLCS_length + 1
        else:
            curr_col = curr_col-1
            
    max_clcs_length = max(max_clcs_length, curr_path.CLCS_length)
#    print "max_clcs_length: ", max_clcs_length
    return curr_path


def calculate_bounds(p_l, p_u, n, i):
    # some bounds may not be populated for some paths
    # for example, p0 won't have a bound for
    # i = m+1
    left_bound = 0
    right_bound = n
#    print "i: ", i
#    print "start: ", p_u.start
#    print "end: ", p_l.end
#    print "p right bounds: ", p_l.bounds[i]
#    print "p left bounds: ", p_u.bounds[i]

    if np.isinf(abs(p_u.u_bounds[i])):
        left_bound = 0
    else:
        left_bound = p_u.u_bounds[i]
        
    if np.isinf(abs(p_l.l_bounds[i])):
        right_bound = n
    else:
        right_bound = p_l.l_bounds[i]
    return (left_bound, right_bound)
    
def single_shortest_path(A,B,mid,p_l,p_u):
        global lcs_arr
        global dp_arr
        m = len(A)
        n = len(B)
        A = A + A
        lcs_arr[mid:mid+m,0:n] = np.zeros((m,n), dtype=int)
        dp_arr[mid:mid+m,0:n] = np.zeros((m,n), dtype=int)
#        for i in range(mid, mid+m+1):
#            #left_bound, right_bound = calculate_bounds(p_l, p_u, n, i)
#            for j in range(0, n+1):
#                lcs_arr[i][j] = 0
#                dp_arr[i][j] = 0
                
#        lcs_arr = np.zeros((4096, 2048), dtype=int)
#        dp_arr = np.zeros((4096, 2048), dtype=int)
#        print p_l.l_bounds[mid:mid+m+1]
#        print p_u.u_bounds[mid:mid+m+1]

        # find a path within the bounds
        for i in range(mid+1, mid+m+1):
            left_bound, right_bound = calculate_bounds(p_l, p_u, n, i)
            if left_bound == 0:
                left_bound = 1
            for j in range(int(left_bound), int(right_bound) + 1):
                if A[i-1] == B[j-1] and can_move_diag(mid, i, j, n, p_l, p_u):
                    lcs_arr[i][j] = lcs_arr[i-1][j-1]+1
                    dp_arr[i][j] = DIAG
                elif lcs_arr[i-1][j] > lcs_arr[i][j-1] and \
                     can_move_up(mid, i, j, n, p_l, p_u):
                    # up
                    lcs_arr[i][j] = lcs_arr[i-1][j]
                    dp_arr[i][j] = UP
	        else:
                    if can_move_left(mid, i, j, n, p_l, p_u):
		        lcs_arr[i][j] = lcs_arr[i][j-1]
                        # left
                        dp_arr[i][j] = LEFT
        #print_matrix(lcs_arr,2*m+1,n+1)
        #sys.exit()
        return retrace_recursive(mid, mid+m, n, p_l, p_u)

def can_move_diag(start_row, curr_row, curr_col, n, p_l, p_u):
    if curr_row-1 < start_row:
        return False
    left_bound, right_bound = calculate_bounds(p_l, p_u, n, curr_row-1)
    if curr_col - 1 > right_bound:
        return False
    if curr_col - 1 < left_bound:
        return False
    if curr_col - 1 < 0:
        return False
    return True

def can_move_up(start_row, curr_row, curr_col, n, p_l, p_u):
    if curr_row-1 < start_row:
        return False
    left_bound, right_bound = calculate_bounds(p_l, p_u, n, curr_row-1)
    if curr_col > right_bound:
        return False
    # not sure if we'll ever get here...
    if curr_col < left_bound:
        return False
    return True

def can_move_left(start_row, curr_row, curr_col, n, p_l, p_u):
    if curr_col-1 < 0:
        return False
    left_bound, right_bound = calculate_bounds(p_l, p_u, n, curr_row)
    if curr_col-1 < left_bound:
        return False
    return True
        
def retrace_recursive(start_row, end_row, end_col, p_l, p_u):
    # starting at the last node of the array, trace back
    # a shortest path
    global max_clcs_length
    curr_path = Path()
    
    curr_row = end_row
    curr_col = end_col
    
#    for i in range(start_row + 1, end_row + 1):
#        dp_arr[i][0] = UP
#    for j in range(1, end_col + 1):
#        dp_arr[start_row][i] = LEFT
#    print_matrix(lcs_arr,2*4+1,5+1)    
        
    # initialize paths as a set of bounds
    while True:
        # the min only really matters if we take a left
        curr_path.l_bounds[curr_row] = max(curr_path.l_bounds[curr_row],
                                                curr_col)
        curr_path.u_bounds[curr_row] = min(curr_path.u_bounds[curr_row],
                                                curr_col)

        # find a path to the top left corner of the block
        # the rows will start at different places (0-m)
        # but the ending column will always be 0
        if curr_row == start_row and curr_col == 0:
            break

        if curr_row == start_row and \
           can_move_left(start_row, curr_row, curr_col, end_col, p_l, p_u):
            curr_col = curr_col - 1
        elif curr_col == 0 and \
             can_move_up(start_row, curr_row, curr_col, end_col, p_l, p_u):
            curr_row = curr_row - 1
        elif dp_arr[curr_row][curr_col] == DIAG:
            curr_row = curr_row-1
            curr_col = curr_col-1
            curr_path.CLCS_length = curr_path.CLCS_length + 1
        elif dp_arr[curr_row][curr_col] == LEFT:
            curr_col = curr_col-1
        # move up
        else:
            curr_row = curr_row-1
#    if start_row == 3:
#        print "LENGTH: ", curr_path.CLCS_length
    max_clcs_length = max(max_clcs_length, curr_path.CLCS_length)
#    print "max_clcs_length: ", max_clcs_length
    return curr_path

def print_matrix(matrix, l1, l2):
        for i in range(l1):
                for j in range(l2):
                        sys.stdout.write(str(matrix[i,j]))
                sys.stdout.write("\n")
        sys.stdout.write("\n\n")
        for i in range(l1):
                for j in range(l2):
                        sys.stdout.write(str(dp_arr[i,j]))
                sys.stdout.write("\n")

def main():
        global max_clcs_length
        global lcs_arr
        global dp_arr
        global paths
        
	if len(sys.argv) != 1:
	    sys.exit('Usage: `python LCS.py < input`')
	
	for l in sys.stdin:
		A,B = l.split()
                compute_p0_pm(A,B)
                find_shortest_paths(A,B,paths,0,len(A))
                print max_clcs_length
#                print paths[len(A)/2].u_bounds[0:2*len(A)]
#                print paths[len(A)/2].l_bounds[0:2*len(A)]
#                return
                lcs_arr = np.zeros((4096, 2048), dtype=int)
                dp_arr = np.zeros((4096, 2048), dtype=int)
                paths = np.empty(2048, dtype=object)
                max_clcs_length = 0
	return

if __name__ == '__main__':
	main()
