using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MathHelper
{
    public static readonly double Eps = 10e-5;
    public static bool InfLineIntersection(Vector2 l1p1, Vector2 l1p2, Vector2 l2p1, Vector2 l2p2, out Vector2 intersection)
    {
        intersection = Vector2.zero;
        float a1 = l1p2.y - l1p1.y;
        float b1 = l1p1.x - l1p2.x;
        float a2 = l2p2.y - l2p1.y;
        float b2 = l2p1.x - l2p2.x;
        float det = a1 * b2 - a2 * b1; //determinant (denominator)

        if (det == 0f) return false;//coincidence or parallel, two segments on the same line
        float c1 = a1 * l1p1.x + b1 * l1p1.y;
        float c2 = a2 * l2p1.x + b2 * l2p1.y;
        float det1 = c1 * b2 - c2 * b1; //determinant (numerator 2)
        float det2 = a1 * c2 - a2 * c1; //determinant (numerator 1)
        intersection.x = det1 / det;
        intersection.y = det2 / det;
        return true;
    }

    public static float TriangularInvLerp(float from, float to, float value)
    {
        var mid = 0.5f * (from + to);
        var firstHalf = Mathf.InverseLerp(from, mid, value);
        if (firstHalf <= 0f)
            return 0f;
        else if (firstHalf >= 1f)
            return 1f - Mathf.InverseLerp(mid, to, value);
        else return firstHalf;
    }
}

public struct Particle
{
    public float mass;
    public float inv_density;
    public Vector3 position;
    public Vector3 velocity;
    public int onSurface;
    public Vector3 midVelocity;
    public Vector3 prevVelocity;
    public float pressure;
    public Vector3 forcePressure;
    public Vector3 forceViscosity;
    public Vector3 forceTension;
    public int cellIdx1d;

    public Particle(float mass, float inv_density, Vector3 position)
    {
        this.mass = mass;
        this.inv_density = inv_density;
        this.position = position;
        this.velocity = Vector3.zero;
        this.onSurface = 0;
        this.midVelocity = Vector3.zero;
        this.prevVelocity = Vector3.zero;
        this.pressure = 0f;
        this.forcePressure = Vector3.zero;
        this.forceViscosity = Vector3.zero;
        this.forceTension = Vector3.zero;
        this.cellIdx1d = 0;
    }

    public Particle(float mass, Vector3 position, Vector3 velocity)
    {
        this.mass = mass;
        this.inv_density = 0f;
        this.position = position;
        this.velocity = velocity;
        this.onSurface = 0;
        this.midVelocity = Vector3.zero;
        this.prevVelocity = Vector3.zero;
        this.pressure = 0f;
        this.forcePressure = Vector3.zero;
        this.forceViscosity = Vector3.zero;
        this.forceTension = Vector3.zero;
        this.cellIdx1d = 0;
    }

    public static int stride = sizeof(float) * 27 + sizeof(int) * 2; // bool is 4 bytes on GPU
}

public class ParticleComparer : IComparer<Particle>
{
    public int Compare(Particle x, Particle y)
    {
        if (x.cellIdx1d < y.cellIdx1d)
            return -1;
        else if (x.cellIdx1d > y.cellIdx1d)
            return 1;
        else return 0;
    }

    public static ParticleComparer comparerInst = new ParticleComparer();
}
