#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

class Color
{
public:
    Color() {}
    Color( unsigned char rbyte, unsigned char gbyte, unsigned char bbyte )
        : r(rbyte / 255.0f), g(gbyte / 255.0f), b(bbyte / 255.0f) {}
    Color shadow( float shadow )
    {
        Color c;
        c.r = r - shadow; if ( c.r < 0.0f ) c.r = 0.0f;
        c.g = g - shadow; if ( c.g < 0.0f ) c.g = 0.0f;
        c.b = b - shadow; if ( c.b < 0.0f ) c.b = 0.0f;
        return c;
    }
	float r;
	float g;
	float b;
};

class ColorRange
{
public:
    ColorRange(): lower(Color(59,76,192)), upper(Color(180,6,38)) {}
    ColorRange( Color l, Color u ): lower(l), upper(u) {}
    Color get( float rIndex )
    {
        Color c;
        c.r = computeChannelValue(lower.r, upper.r, rIndex);
        c.g = computeChannelValue(lower.g, upper.g, rIndex);
        c.b = computeChannelValue(lower.b, upper.b, rIndex);
        return c;
    }
private:
    Color lower;
    Color upper;

    static float computeChannelValue( float chval_a, float chval_b, float rIndex )
    {
        float chmin=chval_a, chmax=chval_b;
        if ( chval_a > chval_b ) { chmin = chval_a; chmax = chval_b; }
        return chmin + ((chmax - chmin) * rIndex);
    }
};

#endif