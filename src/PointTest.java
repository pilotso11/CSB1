/**
 * Created by Seth on 11/13/2016.
 */
public class PointTest
{
    public static void main(String args[])
    {

        Point zero = new Point(0,0);
        Point p5000 = new Point(5000, 0);

        Point r0 = p5000.rotate(zero, 0);
        assert(r0.x == p5000.x && r0.y == p5000.y);

        r0 = p5000.rotate(zero, 90);
        assert(r0.x == p5000.y && r0.y == p5000.x);

        int angle = p5000.getAngle(zero, r0);
        assert(angle == 90);
    }

    static class Point
    {
        int x, y;
        Point(int x, int y)
        {
            this.x = x;
            this.y=y;
        }
        Point(Point a)
        {
            x = a.x;
            y = a.y;
        }

        Point sub(Point b)
        {
            return new Point(x-b.x, y-b.y);
        }
        Point add(Point b)
        {
            return new Point(x+b.x, y+b.y);
        }

        Point mul(float scale)
        {
            return new Point((int) (x * scale), (int) (y * scale));
        }

        Point rotate(Point origin, float degrees)
        {
            double n = Math.toRadians(degrees);

            Point zeroed = this.sub(origin);

            double rx = (zeroed.x * Math.cos(n)) - (zeroed.y * Math.sin(n));
            double ry = (zeroed.x * Math.sin(n)) + (zeroed.y * Math.cos(n));

            return new Point((int) rx, (int) ry).add(origin);
        }

        int getAngle(Point origin, Point second)
        {
            Point zeroed1 = this.sub(origin);
            Point zeroed2 = second.sub(origin);

            double r = Math.atan2(zeroed2.y, zeroed2.x) - Math.atan2(zeroed1.y, zeroed1.x);

            return (int) Math.toDegrees(r);
        }

        public String toString()
        {
            return "(" + x + "," + y + ")";
        }
    }
}
