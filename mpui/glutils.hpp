#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#define COLOR_RANGE_LOWER  0.231373f, 0.298039f,  0.752941f
#define COLOR_RANGE_MIDDLE 0.865003f, 0.865003f,  0.865003f
#define COLOR_RANGE_UPPER  0.705882f, 0.0156863f, 0.14902f

class Color
{
public:
    Color() {}
    Color( float r, float g, float b ): r(r), g(g), b(b) {}
    Color( unsigned char rbyte, unsigned char gbyte, unsigned char bbyte )
        : r(rbyte / 255.0f), g(gbyte / 255.0f), b(bbyte / 255.0f) {}
    Color( const Color &c ): r(c.r), g(c.g), b(c.b) {}
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
    ColorRange(): lower(COLOR_RANGE_LOWER), middle(COLOR_RANGE_MIDDLE), upper(COLOR_RANGE_UPPER) {}
    ColorRange( Color l, Color u ): lower(l), upper(u) {}
    Color get( float rIndex )
    {
        Color l, u, c;
        if ( rIndex < 0.5f ) { l = lower; u = middle; rIndex /= 0.5f; } 
        else               { l = middle; u = upper; rIndex = (rIndex-0.5f)/0.5f; } 
        c.r = computeChannelValue(l.r, u.r, rIndex);
        c.g = computeChannelValue(l.g, u.g, rIndex);
        c.b = computeChannelValue(l.b, u.b, rIndex);
        return c;
    }
private:
    Color lower;
    Color middle;
    Color upper;

    static float computeChannelValue( float chval_a, float chval_b, float rIndex )
    {
        float chmin=chval_a, chmax=chval_b, sign=1.0f;
        if ( chval_a > chval_b ) { chmin = chval_b; chmax = chval_a; sign=-1.0f; }
        return chval_a + ((chmax - chmin) * rIndex * sign);
    }
};

#endif