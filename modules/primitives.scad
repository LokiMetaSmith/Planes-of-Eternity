// modules/primitives.scad
// Common geometric primitives and helpers

module RoundedBox(size=[10,10,10], radius=1, center=false) {
    x = size[0];
    y = size[1];
    z = size[2];

    // Ensure radius isn't too big for the size
    r = min(radius, min(x, y)/2);

    translate(center ? [-x/2, -y/2, -z/2] : [0,0,0])
    hull() {
        translate([r, r, 0]) cylinder(r=r, h=z, $fn=50);
        translate([x-r, r, 0]) cylinder(r=r, h=z, $fn=50);
        translate([x-r, y-r, 0]) cylinder(r=r, h=z, $fn=50);
        translate([r, y-r, 0]) cylinder(r=r, h=z, $fn=50);
    }
}
