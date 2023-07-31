
W_student = np.array(
     # input                                             # output
     #[fu, lu, nu, ru, fl, nl, nr, fr, ld, nd, rd, fd]   #
    [[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # up
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # down
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # left
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]) # right

# comment out these nested for loops if you don't want use them to set the W
# W values and you'd rather just enter them manually above
for i, output in enumerate(output_struct):
  for j, input in enumerate(percept_struct):
    if input == 'near ' + output:
      W_student[i,j] = 4.0
    elif output in input:
      W_student[i,j] = 1.0

display(W_student)