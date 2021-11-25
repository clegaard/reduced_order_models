// Gmsh project created on Wed Nov 20 10:48:37 2019
//+
// mesh size
h = 0.1;
Point(1) = {0, 0, 0, h};
Point(2) = {1/3, 0, 0, h};
Point(3) = {2/3, 0, 0, h};
Point(4) = {1, 0, 0, h};
Point(5) = {0, 1/3, 0, h/10};//10
Point(6) = {1/3, 1/3, 0, h/5};//5
Point(7) = {2/3, 1/3, 0, h/5};//5
Point(8) = {1, 1/3, 0, h};
Point(9) = {0, 2/3, 0, h/10};//10
Point(10) = {1/3, 2/3, 0, h/5};//5
Point(11) = {2/3, 2/3, 0, h/5};//5
Point(12) = {1, 2/3, 0, h};
Point(13) = {0, 1, 0, h};
Point(14) = {1/3, 1, 0, h};
Point(15) = {2/3, 1, 0, h};
Point(16) = {1, 1, 0, h};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 8};
//+
Line(5) = {8, 12};
//+
Line(6) = {12, 16};
//+
Line(7) = {16, 15};
//+
Line(8) = {15, 14};
//+
Line(9) = {14, 13};
//+
Line(10) = {13, 9};
//+
Line(11) = {9, 5};
//+
Line(12) = {5, 1};
//+
Line(13) = {5, 6};
//+
Line(14) = {6, 7};
//+
Line(15) = {7, 8};
//+
Line(16) = {9, 10};
//+
Line(17) = {10, 11};
//+
Line(18) = {11, 12};
//+
Line(19) = {2, 6};
//+
Line(20) = {6, 10};
//+
Line(21) = {10, 14};
//+
Line(22) = {3, 7};
//+
Line(23) = {7, 11};
//+
Line(24) = {11, 15};
//+
Curve Loop(1) = {1, 19, -13, 12};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 22, -14, -19};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, 4, -15, -22};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, 20, -16, 11};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {14, 23, -17, -20};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {15, 5, -18, -23};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {16, 21, 9, 10};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {17, 24, 8, -21};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {18, 6, 7, -24};
//+
Plane Surface(9) = {9};
//+
Physical Curve("Inlet") = {11};
//+
Physical Curve("Wall") = {12, 1, 2, 3, 10, 9, 8, 7};
//+
Physical Curve("Outlet") = {6, 5, 4};
//+
Physical Surface("Omega1") = {1};
//+
Physical Surface("Omega2") = {2};
//+
Physical Surface("Omega3") = {3};
//+
Physical Surface("Omega4") = {4};
//+
Physical Surface("Omega5") = {5};
//+
Physical Surface("Omega6") = {6};
//+
Physical Surface("Omega7") = {7};
//+
Physical Surface("Omega8") = {8};
//+
Physical Surface("Omega9") = {9};
//+
Physical Surface("All") = {1, 2, 3, 4, 5, 6, 7, 8, 9};