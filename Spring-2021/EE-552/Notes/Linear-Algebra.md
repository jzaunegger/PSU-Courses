# Linear Algebra

This course is intended to supplement a undergraduate linear algebra course. This class is written and taught by Jim Hefferon at St. Michaels College, and released this book for free as well as a youtube video available at https://www.youtube.com/watch?v=JnTa9XtvmfI 

The intention of this course is to watch the video, run through the examples in the book, and use book to work through the examples, and complete the questions in the book. There is a seperate book that provides answers.

# Chapter 1 - Solving Linear Systems

A linear combination of x1, ... xn has the form of a1x1 + a2x2 + a3x3 + .... anxn where the a variables, represent the combinations coefficents. In the example below. Here (1/4) is the coeffecient of x, and 1 would be a coeffecient of y and z. It is important to not that coeffecients of 1 are not written, but implied. 

    (1/4)x + y - z

A linear equations is just a linear combination that is equal to some variable which you are trying to solve for.  We want to find the solution that if you were to supplement the x variables with some real numbers, they summate to whatever d is. An example of a linear equation is 

    a1,2 X1 + a1,2 X2 + a1,3 X3 + ... a1,n Xn = d1

As opposed to a system of linear equations is  

    a1,2 X1 + a1,2 X2 + a1,3 X3 + ... a1,n Xn = d1
    a1,2 X1 + a1,2 X2 + a1,3 X3 + ... a1,n Xn = d2
                        ...
    am,2 X1 + am,2 X2 + am,3 X3 + ... a1,n Xn = dm

where the goal or solution is a n sized tuple of solutions for all equations in the system. 

A practical example of this is:

    (1/4)x + y - z = 0
       x + 4y + 2z = 12
       2x - 3y - z = 3

A method of finding the solution that works every time and is the fastest known method, is known as gauss's method (Linear Elimination). This method wants us to transform the system, into something that is easier to solve. 

Step 1: 4r1 or in other words multiply the coeffecients of row 1 by 4

    x + 4y - 4z = 0
    x + 4y + 2z = 12
    3x -3y - z = 3

Step 2: -r1+r2 and -2r1 + r3 or take the coeffeients of row 1, make them negative and add them to the coeffecients of row 2. Then take the coeffecients of row 1 and multiply them by negative 2 and add them to the coeffecients of row 3. Notice we only have row 1 is the only one that contains a row.

    x + 4y - 4z = 0
             6z = 12
      -11y + 7z = 3

Step 3: r2 <-> r3: Swap row 2 and row 3. So far all we have done is transform this system in a way that makes it easy to see what z equals. Z = 12/6 or 2. Now that we know z, we can then calculate y , and after we have y we can determine x, allowing us to solve this system. 

    x + 4y - 4z = 0
      -11y + 7z = 3
            6z = 12

Using the steps above, we can determine the value of x to equal 4, y to equal 1, and z to equal 2. We can prove this by returning to the original equation and checking our solved values, to ensure our process is correct. If this process is done correctly, the answer will never be wrong. 

In each row of the system, the first variable with a non-zero coeffecient is called the leading variable. So in our case for row 1 it would be (1/4)y, in row 2 4y, and row 3 is 2x. (From the original system) The system is in echelon form is each leading variable is to the right of the leading variable in the row above it, except for the leading variable in row 1, and the rows with all-zero coeffecients are at the bottom. So in the case of our example, the system after step 3 is in the echelon form.

Example 2:

    2x - 3y - z + 2w = -2
         x + 3z + 1w = 6
    2x - 3y - z + 3w = -3
          y + z - 2w = 4

If we use the following transormations, (1/2)r3 + r2 and -r1 + r3, we get the result

    2x - 3y - z + 2w = -2
     (3/2)y + (7/2)z = 7
                   w = -1
          y + z - 2w = 4

We can then apply another transformation to isolate Y. (-2/3)r2 + r4 which gives us a result of

    2x - 3y - z + 2w = -2
     (3/2)y + (7/2)z = 7
                   w = -1
        -(4/3)z - 2w = -2/3 

Then r3 <-> r4, results in

    2x - 3y - z + 2w = -2
     (3/2)y + (7/2)z = 7
        -(4/3)z - 2w = -2/3 
                   w = -1


Now that we have transformed this system, we can use w to solve for z. Using w and z, we can solve for y, and then use w, z, and y to solve for x. Giving us our answer that z = -1, z = 2, y = 0, and x = 1, which can be expressed as the tuple (1, 0, 2, -1)

Returning to Gauss's Method, this theorm states that if a linear system is changed by swapping rows, has multiplied both sides by some non-zero constant, or the equations is replaced by the sum of itself and the multiple of another, than the two systems have the same set of solutions. This operations are known as Gaussian operations
    1. Swapping
    2. Multiplying by a scalar (Rescaling)
    3. Row Combination

LEFT OFF ABOUT 25 MINUTES IN.