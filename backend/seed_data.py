"""Expanded Seed generator for 10 Official University Regulations and FAISS Vector Database"""
import os
import json
import numpy as np
from app.document_processor import process_document, FastSemanticEmbedder
from app.vector_store import VectorStore

DOCUMENTS_DATA = [
    {
        "id": "DOC001",
        "title": "Academic Regulations & Curriculum Guidelines",
        "filename": "academic_regulations.txt",
        "category": "Academic",
        "source_url": "https://university.edu/regulations/academic_2025.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL UNIVERSITY ACADEMIC REGULATIONS (R-2025)
SECTION 1: GRADING SYSTEM AND CGPA CALCULATION
1.1 Letter Grades and Grade Points:
- O (Outstanding): Grade Points = 10 (Marks Range: 90 - 100%)
- A+ (Excellent): Grade Points = 9 (Marks Range: 80 - 89%)
- A (Very Good): Grade Points = 8 (Marks Range: 70 - 79%)
- B+ (Good): Grade Points = 7 (Marks Range: 60 - 69%)
- B (Above Average): Grade Points = 6 (Marks Range: 50 - 59%)
- C (Pass): Grade Points = 5 (Marks Range: 40 - 49%)
- F (Fail / Backlog): Grade Points = 0 (Marks Range: Below 40%)
- Ab (Absent): Grade Points = 0

1.2 CGPA and SGPA Formulas:
The Semester Grade Point Average (SGPA) is calculated as SGPA = Sum(Credit_i * GradePoint_i) / Sum(Credit_i).
The Cumulative Grade Point Average (CGPA) is computed as CGPA = Sum(All_Credits_Earned * GradePoints) / Total_Credits_Attempted.
A minimum CGPA of 5.0 is strictly required for the award of an Undergraduate Degree.
Total credits required for a 4-year B.Tech degree: 160 credits.

=== PAGE 2 ===
SECTION 2: COURSE REGISTRATION, PROBATION, AND DURATION
2.1 Add/Drop Policy:
Students may add or drop elective courses within the first 14 calendar days (2 weeks) of the semester commencement with advisor approval.
Audit courses carry zero credits and do not affect the CGPA calculation.

2.2 Academic Probation:
Any student whose CGPA falls below 5.0 at the end of an academic year will be placed on Academic Probation for the subsequent semester.
Students on probation may not register for more than 18 credits in that semester and must attend weekly mandatory mentoring sessions.

2.3 Maximum Duration for Degree Completion:
The maximum duration permitted for completing a 4-year undergraduate degree is N + 2 years (i.e., 6 consecutive academic years).
Failure to complete degree requirements within 6 years will result in termination of student enrollment.

=== PAGE 3 ===
SECTION 3: HONORS AND MINOR DEGREE PROGRAMS
3.1 Eligibility for Honors Degree:
Students with a CGPA of 7.50 or higher at the end of the 4th semester with no standing backlogs are eligible to register for an Honors Degree in their parent discipline.
Students must earn an additional 18 to 20 specialized credits over and above the standard 160 degree credits.

3.2 Minor Degree in Interdisciplinary Fields:
Students can earn a Minor Degree (e.g., AI & Machine Learning, Data Science, Fintech) by completing 18 designated credits from another department without exceeding a maximum term load of 26 credits per semester."""
    },
    {
        "id": "DOC002",
        "title": "Examination Regulations & Assessment Rules",
        "filename": "examination_rules.txt",
        "category": "Examination",
        "source_url": "https://university.edu/exams/examination_manual.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL UNIVERSITY EXAMINATION MANUAL
SECTION 1: ASSESSMENT WEIGHTAGE AND ELIGIBILITY
1.1 Evaluation Structure:
The assessment in each theory course comprises:
- Continuous Internal Evaluation (CIE): 40% weightage (Mid-term tests: 25%, Quizzes/Assignments: 10%, Class Participation: 5%).
- Semester End Examination (SEE): 60% weightage.
A candidate must secure at least 40% in SEE and 40% in overall aggregate (CIE + SEE) to pass the course.

1.2 Hall Ticket Eligibility:
Hall tickets for the Semester End Examination are issued only to registered students with:
1. Minimum 75% attendance in the course.
2. Complete clearance of all semester university tuition and laboratory dues.
3. No pending disciplinary suspensions.

=== PAGE 2 ===
SECTION 2: RE-EVALUATION AND SUPPLEMENTARY EXAMINATIONS
2.1 Re-evaluation and Answer Script Inspection:
Students unsatisfied with their Semester End Examination marks may apply for Answer Script Photocopy and Re-evaluation within 15 calendar days from the date of result declaration.
The fee for re-evaluation is $25 (Rs. 500) per course.
If the re-evaluation yields a difference of +/- 10% or more from original marks, the revised score will be updated on the grade card.

2.2 Supplementary / Backlog Exams:
Supplementary examinations for odd and even semester backlogs are conducted twice annually (July and January).
A student can register for a maximum of 4 backlog courses in a supplementary examination cycle.

=== PAGE 3 ===
SECTION 3: MALPRACTICE RULES AND DISCIPLINARY SANCTIONS
3.1 Malpractice Sanctions:
Possession of unauthorized materials, chits, written notes, or programmable calculators: Cancellation of examination in that specific paper.
Possession or use of mobile phones, smartwatches, or Bluetooth earpieces in exam hall: Cancellation of all examinations of that semester and debarment for 1 additional semester.
Impersonation or proxy writing: Immediate expulsion from the university and legal action under the University Public Examinations Act."""
    },
    {
        "id": "DOC003",
        "title": "Student Attendance Policy & Leave Rules",
        "filename": "attendance_policy.txt",
        "category": "Attendance",
        "source_url": "https://university.edu/policies/attendance_policy.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL UNIVERSITY ATTENDANCE POLICY
SECTION 1: MANDATORY ATTENDANCE REQUIREMENTS
1.1 Aggregate Attendance Threshold:
Every regular student is mandated to maintain a minimum of 75% attendance in aggregate across all registered courses (theory and practicals) in every academic semester.
Attendance is calculated from the official semester start date until the last instructional working day.

1.2 Laboratory & Practical Attendance:
Laboratory courses require a strict 80% physical attendance. Missed lab experiments must be completed during designated make-up sessions prior to practical exams.

=== PAGE 2 ===
SECTION 2: CONDONATION AND SHORTAGE CONSEQUENCES
2.1 Condonation of Attendance Shortage (Medical & Sports):
The Dean of Academic Affairs may condone attendance shortage up to 10% (i.e., between 65% and 74.9% attendance) strictly on grounds of:
- Prolonged hospitalization or certified illness (Official medical certificate must be submitted within 3 working days of resuming classes).
- University representation in National / Inter-University sports, cultural, or NCC/NSS events.
No condonation is permitted for students with attendance below 65% under any circumstances.

2.2 Detention Due to Shortage (Grade SA):
Students who fail to secure 75% attendance (or 65% with approved condonation) will be awarded the 'SA' (Shortage of Attendance) grade.
Students with Grade SA are detained from taking the Semester End Examination in that course and must re-register for the course during the subsequent academic year."""
    },
    {
        "id": "DOC004",
        "title": "Student Handbook & Campus Code of Conduct",
        "filename": "student_handbook.txt",
        "category": "Handbook",
        "source_url": "https://university.edu/student-life/handbook.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL STUDENT HANDBOOK & CODE OF CONDUCT
SECTION 1: CAMPUS ETHICS AND ANTI-RAGGING DIRECTIVE
1.1 Anti-Ragging Directives (Zero Tolerance):
Ragging in any form (physical, psychological, verbal, or online harassment) is strictly prohibited on campus, hostels, and transport facilities.
Any student found guilty of ragging faces immediate suspension, expulsion from the university, and criminal FIR lodging under State Anti-Ragging Acts.
Anti-Ragging 24/7 Helpline: 1800-180-5522.

1.2 Grievance Redressal and Internal Complaints Committee:
Students facing discrimination, unfair grading, or harassment may file complaints through the online Student Grievance Portal or directly to the Proctorial Board.
All grievances are investigated confidentially within 7 working days.

=== PAGE 2 ===
SECTION 2: HOSTEL REGULATIONS AND LIBRARY USAGE
2.1 Hostel Curfew and Resident Guidelines:
Hostel gate closing time is strictly 10:00 PM on weekdays (Monday - Friday) and 10:30 PM on weekends.
Biometric attendance is recorded nightly at 10:15 PM.
Day scholars and unauthorized guests are strictly prohibited inside hostel rooms past 7:00 PM.

2.2 Central Library Guidelines:
Undergraduate students may borrow up to 4 books simultaneously for a duration of 14 calendar days.
Overdue books incur a fine of $0.50 (Rs. 10) per day per book.
E-library databases (IEEE, ACM, ScienceDirect) are accessible 24/7 via university VPN login."""
    },
    {
        "id": "DOC005",
        "title": "Scholarship Guidelines & Merit Waivers",
        "filename": "scholarship_guidelines.txt",
        "category": "Scholarship",
        "source_url": "https://university.edu/admissions/scholarships_fees.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL UNIVERSITY SCHOLARSHIP REGULATIONS
SECTION 1: MERIT AND NEED-BASED SCHOLARSHIPS
1.1 Chancellor's Merit Scholarship:
- Top 5% rank holders with CGPA >= 9.20 receive a 50% tuition fee waiver for the following academic year.
- Students maintaining CGPA between 8.50 and 9.19 with zero backlogs receive a 25% tuition fee waiver.
- Sports Excellence Scholarship: 100% tuition waiver for medalists at National / All-India Inter-University Games.

1.2 Need-Based Financial Assistance:
Full or partial tuition concessions are provided to students whose total parental annual income is below $4,000 (Rs. 3.0 Lakhs) upon submission of valid income certificates.

=== PAGE 2 ===
SECTION 2: CONTINUITY OF SCHOLARSHIP
2.1 Renewal Criteria:
All scholarship recipients must maintain a minimum SGPA of 8.00 in each semester without any backlogs or disciplinary warnings.
Failure to maintain minimum GPA results in forfeiture of the scholarship for subsequent terms."""
    },
    {
        "id": "DOC006",
        "title": "Internship, Placement & Industrial Training Policy",
        "filename": "internship_policy.txt",
        "category": "Academic",
        "source_url": "https://university.edu/career/internship_policy.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL INTERNSHIP & CAREER SERVICES REGULATIONS
SECTION 1: INDUSTRIAL INTERNSHIP REQUIREMENTS
1.1 Mandatory Summer Internships:
All undergraduate engineering and management students must undergo a mandatory minimum of 6 weeks of industrial training or internship between the 6th and 7th semesters.
The internship carries 4 academic credits upon submission of the project report and company evaluation letter.

1.2 Final Semester Capstone Internship:
Students with a CGPA >= 8.0 and no active backlogs may pursue a full 8th-semester industry internship (20-24 weeks) in lieu of on-campus electives.
Bi-weekly progress logs must be signed by the industry mentor and faculty coordinator.

=== PAGE 2 ===
SECTION 2: CAMPUS PLACEMENT ELIGIBILITY & CONDUCT
2.1 One Student One Job Policy:
Once a student receives a verified dream-tier job offer (CTC > $15,000 / Rs. 10 LPA), they are de-listed from further standard campus placement drives to ensure equal opportunities.
Rejecting an accepted on-campus offer without prior Dean approval results in permanent debarment from career services."""
    },
    {
        "id": "DOC007",
        "title": "Hostel Residence, Dining & Security Bylaws",
        "filename": "hostel_bylaws.txt",
        "category": "Handbook",
        "source_url": "https://university.edu/campus-life/hostel_bylaws.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL RESIDENCE HALL & DINING REGULATIONS
SECTION 1: ALLOCATION AND ENTRY GUIDELINES
1.1 Room Allocation & Inventory:
Hostel rooms are allotted on a yearly basis. Students are responsible for furniture, fittings, and room fixtures.
Damage to property will be deducted from the security caution deposit.

1.2 Night Outpass and Gate Timings:
Students intending to stay out overnight must submit an online e-Outpass verified via SMS/Email by parents at least 6 hours in advance.
Hostel entry after 10:00 PM requires warden clearance. Unauthorized absence incurs a fine of $20 per night.

=== PAGE 2 ===
SECTION 2: DINING HALL AND ELECTRICAL APPLIANCES
2.1 Prohibited Appliances:
Heavy electrical appliances (electric heaters, induction stoves, immersion rods) are strictly prohibited inside hostel rooms for fire safety.
Possession leads to confiscation and a fine of $50.

2.2 Mess Rebate Policy:
Students away from hostel for 5 or more consecutive days on approved academic or medical leave are eligible for a mess fee rebate upon submitting proof to the mess warden."""
    },
    {
        "id": "DOC008",
        "title": "Research Ethics, Plagiarism & IP Policy",
        "filename": "research_ethics.txt",
        "category": "Academic",
        "source_url": "https://university.edu/research/ethics_policy.pdf",
        "content": """=== PAGE 1 ===
ACADEMIC INTEGRITY AND PLAGIARISM BYLAWS
SECTION 1: SIMILARITY THRESHOLDS & THESIS VERIFICATION
1.1 Permissible Similarity Index:
All undergraduate capstone theses, master's dissertations, and doctoral research must be screened through the university Turnitin/Urkund portal.
- Overall Similarity Index must not exceed 10% (excluding bibliography and generic quotes).
- Individual source similarity must be less than 2%.

1.2 Penalties for Plagiarism:
- Level 1 (Similarity 10% - 40%): Student must revise and resubmit the thesis within 6 months.
- Level 2 (Similarity 40% - 60%): Resubmission debarred for 1 full academic year.
- Level 3 (Similarity > 60%): Permanent cancellation of research registration and degree denial.

=== PAGE 2 ===
SECTION 2: INTELLECTUAL PROPERTY & PATENTS
2.1 Student Inventions & Patents:
Inventions developed using substantial university lab infrastructure are jointly owned by the university and student inventors (60:40 revenue sharing model).
The university Technology Transfer Office covers 100% of the patent filing fees for approved innovations."""
    },
    {
        "id": "DOC009",
        "title": "Sports, Cultural Quota & Extra-Curricular Regulations",
        "filename": "sports_regulations.txt",
        "category": "Handbook",
        "source_url": "https://university.edu/sports/regulations.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL SPORTS AND EXTRA-CURRICULAR BYLAWS
SECTION 1: ATTENDANCE & GRACE MARKS FOR ATHLETES
1.1 On-Duty (OD) Attendance Benefit:
Students officially representing the university in AIU (Association of Indian Universities), National, or International sports/cultural meets receive 100% On-Duty (OD) attendance credit for all missed class hours.

1.2 Academic Grace Marks for Medallists:
- Gold Medal at All-India Inter-University Tournament: 5% grace marks in semester aggregate.
- Silver/Bronze Medal: 3% grace marks in semester aggregate.
- Participation in National Championship: 2% grace marks.

=== PAGE 2 ===
SECTION 2: SPORTS SCHOLARSHIPS AND GYM FACILITIES
2.1 Gymnasium and Sports Complex Hours:
The Olympic-size swimming pool and indoor gymnasium are open from 5:30 AM to 8:30 AM and 4:30 PM to 8:30 PM daily.
Valid student identity card and standard athletic attire are strictly mandatory."""
    },
    {
        "id": "DOC010",
        "title": "Tuition Fee Structure, Payment & Refund Rules",
        "filename": "fee_regulations.txt",
        "category": "Scholarship",
        "source_url": "https://university.edu/finance/tuition_fees.pdf",
        "content": """=== PAGE 1 ===
OFFICIAL TUITION FEE AND PAYMENT REGULATIONS
SECTION 1: PAYMENT SCHEDULE & PENALTIES
1.1 Semester Fee Schedule:
Tuition, lab, and development fees must be deposited before the commencement date of each semester.
A grace period of 10 calendar days is allowed.
- Payments made between Day 11 and Day 25: Late fee fine of $10 (Rs. 200) per day.
- Non-payment after Day 30: Temporary suspension of LMS portal access and course registration.

=== PAGE 2 ===
SECTION 2: STATUTORY FEE REFUND SCHEDULE (UGC MANDATE)
2.1 Refund on Admission Cancellation:
- 100% refund (less $15 processing charge): If formal notice of withdrawal is submitted 15 days or more before class commencement.
- 90% refund: If withdrawal is requested less than 15 days before class commencement.
- 80% refund: Within 15 calendar days after class commencement.
- 50% refund: Between 16 and 30 calendar days after class commencement.
- 0% (No refund): More than 30 calendar days after class commencement."""
    }
]


def seed_database():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    reg_dir = os.path.join(data_dir, "regulations")
    meta_dir = os.path.join(data_dir, "metadata")
    vec_dir = os.path.join(data_dir, "vector_db")

    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(vec_dir, exist_ok=True)

    print("[Seed] 1. Writing regulation files and documents.json registry...")
    
    registry = []
    all_chunks = []
    
    embedder = FastSemanticEmbedder()

    for doc in DOCUMENTS_DATA:
        file_path = os.path.join(reg_dir, doc["filename"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"].strip())

        chunks, total_pages = process_document(
            file_path=file_path,
            title=doc["title"],
            category=doc["category"],
            source_url=doc["source_url"],
            chunk_size=800,
            chunk_overlap=150
        )

        all_chunks.extend(chunks)

        registry.append({
            "id": doc["id"],
            "title": doc["title"],
            "filename": doc["filename"],
            "category": doc["category"],
            "source_url": doc["source_url"],
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2)
        })

    # Save documents.json
    registry_path = os.path.join(meta_dir, "documents.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"[Seed] Saved {len(registry)} documents to {registry_path}")

    # Generate embeddings and build vector DB
    print(f"[Seed] 2. Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts)

    print("[Seed] 3. Building and saving vector database...")
    vstore = VectorStore(db_dir=vec_dir)
    vstore.build_index(all_chunks, embeddings)

    print(f"[Seed] [SUCCESS] Seeded {len(registry)} documents, {len(all_chunks)} chunks with {embeddings.shape[1]}d vectors.")


if __name__ == "__main__":
    seed_database()
