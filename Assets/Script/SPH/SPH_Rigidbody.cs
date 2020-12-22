using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SPH_Rigidbody : MonoBehaviour
{
    public int particleNum { get { return particles.Count; } }
    public List<Particle> particles;
    public Rigidbody m_rigidbody = null;
    public float mass;
    private void Start()
    {
        m_rigidbody = GetComponent<Rigidbody>();
    }
}
