#include "mpui.h"
#include "glutils.hpp"

#include <stdio.h>
#include <string.h>
#include <thread>
#include <mutex>

#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#define GET3D(M, rows, columns, i, j, k) \
	( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

#define FRONT  0
#define BACK   1
#define RIGHT  2
#define LEFT   3
#define TOP    4
#define BOTTOM 5

#define CELLRAD  0.004f
#define CELLSIZE CELLRAD * 2
#define CELL_SHADOW_FACTOR 2

#define WINDOW_TITLE  "MPUI Hub"
#define WINDOW_WIDTH  700
#define WINDOW_HEIGHT 500

float xrot = 0.0f;
float yrot = 0.0f;

float xdiff = 100.0f;
float ydiff = 100.0f;

float origin_x = 0.0f;
float origin_y = 0.0f;
float origin_z = 0.0f;

float zoom      = 10.0f;
float resize_f  = 1.0f;
int   mouseDown = 0;

double    *displaybuff      = nullptr;
int        buff_xsize       = 0;
int        buff_ysize       = 0;
int        buff_zsize       = 0;
double     filter_threshold = 0.0f;
bool       onexit           = false;
std::mutex glock;

const int DRAW_CELL_PATTERN[][4][3] =
{
	{{ -1, -1,  1 }, {  1, -1,  1 }, {  1,  1,  1 }, { -1,  1,  1 }},  // front
	{{ -1, -1, -1 }, { -1,  1, -1 }, {  1,  1, -1 }, {  1, -1, -1 }},  // back
	{{  1, -1, -1 }, {  1,  1, -1 }, {  1,  1,  1 }, {  1, -1,  1 }},  // right
	{{ -1, -1,  1 }, { -1,  1,  1 }, { -1,  1, -1 }, { -1, -1, -1 }},  // left
	{{ -1,  1,  1 }, {  1,  1,  1 }, {  1,  1, -1 }, { -1,  1, -1 }},  // top
	{{ -1, -1,  1 }, { -1, -1, -1 }, {  1, -1, -1 }, {  1, -1,  1 }}   // bottom
};
const float CELL_SHADOW[] =
{
	0.04f,  // front
	0.02f,  // back
	0.03f,  // right
	0.01f,  // left
	0.00f,  // top
	0.05f,  // bottom
};


void drawCell( int i, int j, int k, Color color, bool hiddenNeHood[] )
{
	float offset_i = CELLSIZE*i    + origin_x;
	float offset_j = CELLSIZE*(-k) + origin_y;
	float offset_k = CELLSIZE*j    + origin_z;

	if ( !(buff_xsize & 1) ) offset_i += CELLRAD;
	if ( !(buff_ysize & 1) ) offset_k += CELLRAD;
	if ( !(buff_zsize & 1) ) offset_j -= CELLRAD;

	glTranslatef(offset_i, offset_j, offset_k);
	glBegin(GL_QUADS);

	for ( unsigned short side=0; side < 6; ++side )
		if ( hiddenNeHood[side] )
		{
			Color c = color.shadow( CELL_SHADOW[side] * CELL_SHADOW_FACTOR );
			glColor3f(c.r, c.g, c.b);
			for ( unsigned short vertex=0; vertex < 4; ++vertex )
				glVertex3f( DRAW_CELL_PATTERN[side][vertex][0] * CELLRAD,
				            DRAW_CELL_PATTERN[side][vertex][1] * CELLRAD,
							DRAW_CELL_PATTERN[side][vertex][2] * CELLRAD );
		}
	
	glEnd();
	glTranslatef(-offset_i, -offset_j, -offset_k);
}

bool hiddenCell( double cellValue )
{
	return cellValue <= filter_threshold;
}

bool hiddenCell( int i, int j, int k )
{
	if ( i < 0 || j < 0 || k < 0 ||
	     i >= buff_xsize || j >= buff_ysize || k >= buff_zsize )
		return true;
	return hiddenCell( GET3D(displaybuff, buff_xsize, buff_ysize, i, j, k) );
}

void drawLimits()
{
	float domrad_x = (buff_xsize / 2) * CELLSIZE;
	float domrad_y = (buff_zsize / 2) * CELLSIZE;
	float domrad_z = (buff_ysize / 2) * CELLSIZE;

	if ( buff_xsize & 1 ) domrad_x += CELLRAD;
	if ( buff_ysize & 1 ) domrad_z += CELLRAD;
	if ( buff_zsize & 1 ) domrad_y += CELLRAD;

	glTranslatef(origin_x, origin_y, origin_z);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 0.0f);

	// top limits
	glVertex3f(-domrad_x, domrad_y, -domrad_z); glVertex3f( domrad_x, domrad_y, -domrad_z);
	glVertex3f( domrad_x, domrad_y, -domrad_z); glVertex3f( domrad_x, domrad_y,  domrad_z);
	glVertex3f( domrad_x, domrad_y,  domrad_z); glVertex3f(-domrad_x, domrad_y,  domrad_z);
	glVertex3f(-domrad_x, domrad_y,  domrad_z); glVertex3f(-domrad_x, domrad_y, -domrad_z);

	// bottom limits
	glVertex3f(-domrad_x, -domrad_y, -domrad_z); glVertex3f( domrad_x, -domrad_y, -domrad_z);
	glVertex3f( domrad_x, -domrad_y, -domrad_z); glVertex3f( domrad_x, -domrad_y,  domrad_z);
	glVertex3f( domrad_x, -domrad_y,  domrad_z); glVertex3f(-domrad_x, -domrad_y,  domrad_z);
	glVertex3f(-domrad_x, -domrad_y,  domrad_z); glVertex3f(-domrad_x, -domrad_y, -domrad_z);

	// side limits
	glVertex3f(-domrad_x, domrad_y, -domrad_z); glVertex3f(-domrad_x, -domrad_y, -domrad_z);
	glVertex3f( domrad_x, domrad_y, -domrad_z); glVertex3f( domrad_x, -domrad_y, -domrad_z);
	glVertex3f( domrad_x, domrad_y,  domrad_z); glVertex3f( domrad_x, -domrad_y,  domrad_z);
	glVertex3f(-domrad_x, domrad_y,  domrad_z); glVertex3f(-domrad_x, -domrad_y,  domrad_z);

	glEnd();
	glTranslatef(-origin_x, -origin_y, -origin_z);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glLoadIdentity();

	gluLookAt(
	0.0f, 0.0f, 3.0f,
	0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f);

	glRotatef(xrot, 1.0f, 0.0f, 0.0f);
	glRotatef(yrot, 0.0f, 1.0f, 0.0f);
	
	glock.lock();

	if ( displaybuff != nullptr )
	{
		int ibegin = -buff_xsize / 2;
		int iend   = -ibegin;

		int jbegin = -buff_ysize / 2;
		int jend   = -ibegin;

		int kbegin = -buff_zsize / 2;
		int kend   = -kbegin;

		if ( !(buff_xsize & 1) ) --iend;
		if ( !(buff_ysize & 1) ) --jend;
		if ( !(buff_zsize & 1) ) --kend;

		for ( int i=ibegin; i<=iend; ++i )
		for ( int j=jbegin; j<=jend; ++j )
		for ( int k=kbegin; k<=kend; ++k )
		{
			int i_cell=i-ibegin, j_cell=j-jbegin, k_cell=k-kbegin;

			double cellValue = GET3D(displaybuff, buff_xsize, buff_ysize, i_cell, j_cell, k_cell);
			if ( hiddenCell(cellValue) )
				continue;
			
			bool hiddenNeHood[] =
			{
				hiddenCell( i_cell, j_cell+1, k_cell ),  // front
				hiddenCell( i_cell, j_cell-1, k_cell ),  // back
				hiddenCell( i_cell+1, j_cell, k_cell ),  // right
				hiddenCell( i_cell-1, j_cell, k_cell ),  // left
				hiddenCell( i_cell, j_cell, k_cell-1 ),  // top
				hiddenCell( i_cell, j_cell, k_cell+1 )   // bottom
			};

			cellValue += 734.0f;
			double percCellValue = cellValue / 6334.0f;

			ColorRange crange;
			Color color = crange.get(percCellValue);
			
			drawCell(i, j, k, color, hiddenNeHood);
		}
	}

	drawLimits();

	glock.unlock();

	glFlush();
	glutSwapBuffers();
}

void resize(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glViewport(0, 0, w, h);

	gluPerspective(zoom, resize_f * w / h, resize_f, 100 * resize_f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void printHelp()
{
	printf("\n");
	printf("========== MPUI-Hub Help ==========\n");
    printf("W or w : Up\n");
    printf("S or s : Down\n");
    printf("D or d : Right\n");
    printf("A or a : Left\n");
    printf("Z or z : Zoom-In\n");
    printf("X or x : Zoom-Out\n");
	printf("U or u : Rotate Clockwise\n");
    printf("Y or y : Rotate Counterclockwise\n");
}

void keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
    case 'w': case 'W': origin_y += 0.01f; break;
    case 's': case 'S': origin_y -= 0.01f; break;
    case 'a': case 'A': origin_x -= 0.01f; break;
    case 'd': case 'D': origin_x += 0.01f; break;
    case 'u': case 'U': yrot += 1.0f; break;
    case 'y': case 'Y': yrot -= 1.0f; break;
    case 'h': case 'H': printHelp(); break;
    case 'Z': case 'z':
		zoom-=0.1f;
		resize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
		break;
    case 'X': case 'x':
		zoom+=0.1f;
		resize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
		break;
	}

	glock.lock();
	glutPostRedisplay();
	glock.unlock();
}

void mouse(int button, int state, int x, int y)
{
	if ( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN )
	{
		mouseDown = 1;
		xdiff = x - yrot;
		ydiff = -y + xrot;
	}
	else mouseDown = 0;
}

void mouseMotion(int x, int y)
{
	if (mouseDown)
	{
		yrot = x - xdiff;
		xrot = y + ydiff;
		glock.lock();
		glutPostRedisplay();
		glock.unlock();
	}
}

void hubMainLoop()
{
	glutMainLoop();
	glock.lock();
	onexit = true;
	glock.unlock();
}

namespace mpui {

void
MPUI_Hub_init( std::thread *&loopth )
{
	int argc = 0; char *argv[0];
	glutInit(&argc, argv);
	
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow(WINDOW_TITLE);

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(resize);

	// set a background color.
	glClearColor(0.322f, 0.341f, 0.431f, 0.0f);

	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);

	glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS );

	loopth = new std::thread( hubMainLoop );
}

void
MPUI_Hub_finalize( std::thread *&loopth )
{
	loopth->join();
	if ( displaybuff != nullptr )
	{
		delete[] displaybuff;
		displaybuff = nullptr;
	}
	delete loopth;
	loopth = nullptr;
}

void
MPUI_Hub_setWSize( int xsize, int ysize, int zsize )
{
	glock.lock();
	if ( onexit )
	{
		glock.unlock();
		return;
	}
	buff_xsize = xsize;
	buff_ysize = ysize;
	buff_zsize = zsize;
	glutPostRedisplay();
	glock.unlock();
}

void
MPUI_Hub_setBuffer( double *buff, int xsize, int ysize, int zsize )
{
	glock.lock();
	if ( onexit )
	{
		glock.unlock();
		return;
	}
	
	const int buffsize = xsize*ysize*zsize;
	if ( displaybuff != nullptr && buff_xsize*buff_ysize*buff_zsize != buffsize )
		delete[] displaybuff;
	displaybuff = new double[buffsize];
	memcpy(displaybuff, buff, sizeof(double)*buffsize);

	buff_xsize = xsize;
	buff_ysize = ysize;
	buff_zsize = zsize;
	
	glutPostRedisplay();
	glock.unlock();
}

void
MPUI_Hub_filter( double threshold )
{
	glock.lock();
	if ( onexit )
	{
		glock.unlock();
		return;
	}
	filter_threshold = threshold;
	glutPostRedisplay();
	glock.unlock();
}

bool
MPUI_Hub_onexit()
{
	bool ans;
	glock.lock();
	ans = onexit;
	glock.unlock();
	return ans;
}

}