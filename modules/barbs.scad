// modules/barbs.scad

module Barb(d=10, h=20, wall=2) {
    difference() {
        union() {
            cylinder(d=d, h=h, $fn=50);
            // Simple barb rings
            for (i = [0 : 3]) {
                translate([0, 0, h - 5 - i*3])
                cylinder(d1=d+2, d2=d, h=2, $fn=50);
            }
        }
        // Hollow center
        translate([0,0,-0.1])
        cylinder(d=d-wall, h=h+0.2, $fn=50);
    }
}
