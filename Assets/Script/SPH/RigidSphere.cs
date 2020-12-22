using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RigidSphere : SPH_Rigidbody
{
    public float radius;
    private void Start()
    {
        m_rigidbody = GetComponent<Rigidbody>();
        particles.Add(new Particle(mass, transform.position, m_rigidbody.velocity));
    }
}
